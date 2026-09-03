from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as current_date
from pathlib import Path
from typing import Any

from arxiv_digest.delivery.feishu import DeliveryResult, FeishuDelivery
from arxiv_digest.filtering import KeywordFilter
from arxiv_digest.fulltext.reader import FullTextReader
from arxiv_digest.llm.deep_summary import DeepSummarizer
from arxiv_digest.llm.screening import PaperScreener, filter_by_relevance
from arxiv_digest.models import Paper
from arxiv_digest.rendering.markdown import DigestStats, render_digest
from arxiv_digest.selection import DeepReadSelector


DEFAULT_MEMORY_ROOT = Path(__file__).resolve().parent.parent / "memory"


@dataclass(frozen=True)
class PipelineRun:
    date: str
    counts: dict[str, int]
    digest_path: Path | None
    run_path: Path
    errors: list[str]
    delivery: DeliveryResult | None


class DailyPaperPipeline:
    def __init__(
        self,
        *,
        source,
        keyword_filter: KeywordFilter,
        screener: PaperScreener,
        selector: DeepReadSelector,
        fulltext_reader: FullTextReader,
        deep_summarizer: DeepSummarizer,
        topics: list[str | dict[str, Any]],
        relevance_threshold: float = 6,
        daily_limit: int = 50,
        max_screening_candidates: int = 20,
        delivery: FeishuDelivery | None = None,
        memory_root: Path = DEFAULT_MEMORY_ROOT,
    ):
        self.source = source
        self.keyword_filter = keyword_filter
        self.screener = screener
        self.selector = selector
        self.fulltext_reader = fulltext_reader
        self.deep_summarizer = deep_summarizer
        self.topics = topics
        self.relevance_threshold = relevance_threshold
        self.daily_limit = daily_limit
        self.max_screening_candidates = max_screening_candidates
        self.delivery = delivery
        self.memory_root = memory_root

    def run(
        self,
        *,
        date: str = "today",
        max_papers: int | None = None,
        deep_top_k: int | None = None,
        no_deep_read: bool = False,
        no_delivery: bool = False,
        force_screening: bool = False,
        force_deep_summary: bool = False,
        force_delivery: bool = False,
        dry_run: bool = False,
    ) -> PipelineRun:
        run_date = current_date.today().isoformat() if date == "today" else date
        errors: list[str] = []
        statuses: dict[str, str] = {}
        screen_calls_before = self.screener.api_calls
        screen_hits_before = self.screener.cache_hits
        deep_calls_before = self.deep_summarizer.api_calls
        deep_hits_before = self.deep_summarizer.cache_hits

        papers = self.source.list_daily(run_date, max_papers or self.daily_limit)
        if not papers:
            counts = _counts()
            run_path = self._write_run(
                run_date, counts, errors, [], statuses, "not_requested", _llm_stats(
                    self.screener, self.deep_summarizer, screen_calls_before,
                    screen_hits_before, deep_calls_before, deep_hits_before,
                )
            )
            return PipelineRun(run_date, counts, None, run_path, errors, None)

        keyword_matched = self.keyword_filter.filter(papers)
        matched_ids = {_identity(paper) for paper in keyword_matched}
        for paper in papers:
            statuses[_identity(paper)] = (
                "keyword_matched" if _identity(paper) in matched_ids else "keyword_rejected"
            )

        screened = []
        for paper in keyword_matched[: self.max_screening_candidates]:
            try:
                self.screener.screen(paper, force=force_screening)
                screened.append(paper)
                statuses[_identity(paper)] = "screened"
            except Exception as error:
                statuses[_identity(paper)] = "screening_failed"
                errors.append(f"screening {paper.arxiv_id}: {type(error).__name__}")
        for paper in keyword_matched[self.max_screening_candidates :]:
            statuses[_identity(paper)] = "screening_limit"

        accepted = filter_by_relevance(screened, self.relevance_threshold)
        accepted_ids = {_identity(paper) for paper in accepted}
        for paper in screened:
            statuses[_identity(paper)] = (
                "accepted" if _identity(paper) in accepted_ids else "llm_rejected"
            )

        selector = self.selector
        if no_deep_read:
            selector = DeepReadSelector(enabled=False)
        elif deep_top_k is not None:
            selector = DeepReadSelector(
                enabled=selector.enabled,
                top_k=deep_top_k,
                min_relevance=selector.min_relevance,
            )
        selection = selector.select(accepted)

        deep_success = 0
        fulltext_failed = 0
        for paper in selection.deep_read_candidates:
            statuses[_identity(paper)] = "deep_selected"
            fulltext = self.fulltext_reader.read(paper)
            if not fulltext.text:
                fulltext_failed += 1
                paper.deep_summary_status = "fulltext_unavailable"
                statuses[_identity(paper)] = "fulltext_unavailable"
                errors.append(f"fulltext {paper.arxiv_id}: {fulltext.error or 'unavailable'}")
                continue
            summary = self.deep_summarizer.summarize_safe(
                paper, fulltext.text, force=force_deep_summary
            )
            if summary:
                deep_success += 1
                statuses[_identity(paper)] = "deep_summary_success"
            else:
                statuses[_identity(paper)] = "deep_summary_failed"
                errors.append(f"deep summary {paper.arxiv_id}: failed")

        counts = {
            "hf_total": len(papers),
            "keyword_matched": len(keyword_matched),
            "llm_screened": len(screened),
            "accepted": len(accepted),
            "deep_read": len(selection.deep_read_candidates),
            "deep_read_success": deep_success,
        }
        digest_stats = DigestStats(
            hf_total=len(papers),
            keyword_matched=len(keyword_matched),
            llm_accepted=len(accepted),
            deep_read=len(selection.deep_read_candidates),
            keyword_rejected=len(papers) - len(keyword_matched),
            llm_rejected=len(screened) - len(accepted),
            fulltext_failed=fulltext_failed,
        )
        digest = render_digest(
            date=run_date,
            topics=self.topics,
            deep_papers=selection.deep_read_candidates,
            normal_papers=selection.normal_candidates,
            rejected_papers=[paper for paper in papers if _identity(paper) not in accepted_ids],
            stats=digest_stats,
        )
        digest_path = self.memory_root / "digests" / f"{run_date}.md"
        _atomic_write_text(digest_path, digest)

        delivery_result = None
        delivery_status = "not_requested"
        if self.delivery and not no_delivery:
            delivery_result = self.delivery.send(
                date=run_date,
                digest=digest,
                papers=selection.deep_read_candidates + selection.normal_candidates,
                stats=digest_stats,
                force=force_delivery,
                dry_run=dry_run,
            )
            if delivery_result.skipped:
                delivery_status = "skipped_duplicate"
            elif delivery_result.success:
                delivery_status = "dry_run" if dry_run else "success"
            else:
                delivery_status = "delivery_failed"
                errors.append(f"feishu: {delivery_result.error}")

        run_path = self._write_run(
            run_date,
            counts,
            errors,
            papers,
            statuses,
            delivery_status,
            _llm_stats(
                self.screener, self.deep_summarizer, screen_calls_before,
                screen_hits_before, deep_calls_before, deep_hits_before,
            ),
        )
        return PipelineRun(run_date, counts, digest_path, run_path, errors, delivery_result)

    def _write_run(self, date, counts, errors, papers, statuses, delivery_status, llm_stats):
        path = self.memory_root / "runs" / date / "run.json"
        data = {
            "date": date,
            "counts": counts,
            "llm": llm_stats,
            "delivery": {"feishu": delivery_status},
            "errors": errors,
            "papers": [
                {"status": statuses.get(_identity(paper), "collected"), "paper": paper.to_legacy_dict()}
                for paper in papers
            ],
        }
        _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
        return path


def _counts():
    return {
        "hf_total": 0,
        "keyword_matched": 0,
        "llm_screened": 0,
        "accepted": 0,
        "deep_read": 0,
        "deep_read_success": 0,
    }


def _llm_stats(screener, summarizer, screen_calls, screen_hits, deep_calls, deep_hits):
    return {
        "screen_calls": screener.api_calls - screen_calls,
        "screen_cache_hits": screener.cache_hits - screen_hits,
        "deep_calls": summarizer.api_calls - deep_calls,
        "deep_cache_hits": summarizer.cache_hits - deep_hits,
    }


def _identity(paper: Paper):
    return paper.arxiv_id or paper.title


def _atomic_write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
