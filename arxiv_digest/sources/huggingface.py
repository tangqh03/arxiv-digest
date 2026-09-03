from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from datetime import date as current_date
from typing import Any

from huggingface_hub import HfApi

from arxiv_digest.models import Paper
from arxiv_digest.sources.arxiv import ArxivPaperSource


def parse_hf_paper_ids(html: str) -> list[str]:
    ids = re.findall(r"/papers/(\d{4}\.\d+)", html)
    return list(dict.fromkeys(ids))


def _fetch_html(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "arxiv-digest/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def _run_cli(date: str, limit: int) -> str:
    result = subprocess.run(
        [
            "hf",
            "papers",
            "ls",
            "--date",
            date,
            "--limit",
            str(limit),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return result.stdout


class HuggingFacePaperSource:
    def __init__(
        self,
        *,
        api: Any | None = None,
        cli_runner: Callable[[str, int], str] = _run_cli,
        html_fetcher: Callable[[str], str] = _fetch_html,
        arxiv_enricher: Callable[[Iterable[str]], list[Paper]] | None = None,
    ):
        self._api = api or HfApi()
        self._cli_runner = cli_runner
        self._html_fetcher = html_fetcher
        self._arxiv_enricher = arxiv_enricher or ArxivPaperSource().fetch_by_ids

    def list_daily(self, date: str = "today", limit: int = 50) -> list[Paper]:
        resolved_date = current_date.today().isoformat() if date == "today" else date
        if limit <= 0:
            return []

        try:
            records = list(
                self._api.list_daily_papers(
                    date=resolved_date,
                    limit=limit,
                    token=os.environ.get("HF_TOKEN") or False,
                )
            )
        except Exception:
            records = self._list_with_cli_or_html(resolved_date, limit)

        papers = self._deduplicate(_paper_from_hf(record) for record in records)[:limit]
        return self._enrich(papers)

    def _list_with_cli_or_html(self, date: str, limit: int) -> list[Any]:
        try:
            payload = json.loads(self._cli_runner(date, limit))
            if isinstance(payload, dict):
                payload = payload.get("papers", payload.get("items", []))
            if not isinstance(payload, list):
                raise ValueError("HF CLI JSON response must be a list")
            return payload
        except Exception:
            today = current_date.today().isoformat()
            url = (
                "https://huggingface.co/papers"
                if date == today
                else f"https://huggingface.co/papers/date/{date}"
            )
            ids = parse_hf_paper_ids(self._html_fetcher(url))
            return [{"id": paper_id} for paper_id in ids[:limit]]

    @staticmethod
    def _deduplicate(papers: Iterable[Paper]) -> list[Paper]:
        seen = set()
        result = []
        for paper in papers:
            if paper.arxiv_id and paper.arxiv_id not in seen:
                seen.add(paper.arxiv_id)
                result.append(paper)
        return result

    def _enrich(self, papers: list[Paper]) -> list[Paper]:
        if not papers:
            return []
        try:
            enriched = {
                paper.arxiv_id: paper
                for paper in self._arxiv_enricher(p.arxiv_id for p in papers)
            }
        except Exception:
            return papers

        for paper in papers:
            arxiv = enriched.get(paper.arxiv_id)
            if not arxiv:
                continue
            paper.title = paper.title or arxiv.title
            paper.abstract = arxiv.abstract or paper.abstract
            paper.authors = arxiv.authors or paper.authors
            paper.published = arxiv.published or paper.published
            paper.updated = arxiv.updated or paper.updated
            paper.categories = arxiv.categories or paper.categories
            paper.abs_url = arxiv.abs_url or paper.abs_url
            paper.pdf_url = arxiv.pdf_url or paper.pdf_url
        return papers


def _paper_from_hf(record: Any) -> Paper:
    paper_id = _value(record, "id", default="")
    summary = _value(record, "summary", default="") or ""
    return Paper(
        arxiv_id=paper_id,
        title=_value(record, "title", default="") or "",
        authors=[
            _value(author, "name", default="")
            for author in (_value(record, "authors", default=[]) or [])
            if _value(author, "name", default="")
        ],
        abstract=summary,
        published=_iso_date(_value(record, "published_at", "publishedAt")),
        abs_url=f"https://arxiv.org/abs/{paper_id}" if paper_id else None,
        pdf_url=f"https://arxiv.org/pdf/{paper_id}" if paper_id else None,
        sources=["huggingface-daily"],
        hf_url=f"https://huggingface.co/papers/{paper_id}" if paper_id else None,
        hf_upvotes=_value(record, "upvotes"),
        hf_summary=_value(record, "ai_summary", "aiSummary", default=summary),
        hf_keywords=list(_value(record, "ai_keywords", "aiKeywords", default=[]) or []),
        project_url=_value(record, "project_page", "projectPage"),
        github_url=_value(record, "github_repo", "githubRepo"),
    )


def _value(record: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return default


def _iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)[:10] or None
