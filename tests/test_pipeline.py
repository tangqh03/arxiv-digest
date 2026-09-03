import json

from arxiv_digest.delivery.feishu import DeliveryResult
from arxiv_digest.filtering import KeywordFilter
from arxiv_digest.fulltext.reader import FullTextResult
from arxiv_digest.llm.deep_summary import DeepSummarizer
from arxiv_digest.llm.screening import PaperScreener
from arxiv_digest.models import Paper
from arxiv_digest.pipeline import DailyPaperPipeline
from arxiv_digest.selection import DeepReadSelector


class FakeSource:
    def __init__(self, papers):
        self.papers = papers

    def list_daily(self, date, limit):
        return self.papers[:limit]


class FakeLLM:
    model = "pipeline-model"

    def __init__(self, scores=None, fail_id=None):
        self.scores = scores or {}
        self.fail_id = fail_id

    def chat_json(self, messages):
        payload = json.loads(messages[1]["content"])
        paper_id = payload["paper"]["arxiv_id"]
        if paper_id == self.fail_id:
            raise RuntimeError("screen failed")
        relevance = self.scores.get(paper_id, 8)
        return {
            "relevance": relevance,
            "novelty": 7,
            "impact": 7,
            "matched_topic": "Video Reasoning",
            "relevance_reason": "direct match",
            "abstract_zh": "中文摘要",
            "tldr_zh": "简短总结",
            "worth_deep_reading": relevance >= 7,
        }


class FakeFulltext:
    def __init__(self, fail_ids=None):
        self.fail_ids = set(fail_ids or [])
        self.calls = []

    def read(self, paper):
        self.calls.append(paper.arxiv_id)
        if paper.arxiv_id in self.fail_ids:
            return FullTextResult("", None, False, "unavailable")
        return FullTextResult("# Paper\n## Introduction\nfull text", "hf", False, None)


class FakeDeep:
    api_calls = 0
    cache_hits = 0

    def summarize_safe(self, paper, text, force=False):
        self.api_calls += 1
        paper.deep_summary = {
            "one_sentence": "one",
            "problem": {"what": "problem", "why_important": "important"},
            "prior_work_gap": "gap",
            "core_intuition": "intuition",
            "method": {"overview": "overview", "pipeline": ["step"]},
            "experiments": {"setup": "setup", "main_results": ["result"]},
            "limitations": ["limitation"],
            "why_relevant_to_user": "relevant",
            "takeaway": "takeaway",
        }
        paper.deep_summary_status = "success"
        return paper.deep_summary


class FakeDelivery:
    def __init__(self):
        self.calls = 0

    def send(self, **kwargs):
        self.calls += 1
        return DeliveryResult(True, 1)


def make_paper(index, text="video reasoning"):
    return Paper(
        arxiv_id=f"2609.{index:05d}",
        title=f"Paper {index}: {text}",
        abstract=f"An abstract about {text}.",
        abs_url=f"https://arxiv.org/abs/2609.{index:05d}",
        pdf_url=f"https://arxiv.org/pdf/2609.{index:05d}",
        hf_url=f"https://huggingface.co/papers/2609.{index:05d}",
    )


def build_pipeline(tmp_path, papers, *, scores=None, fail_screen=None, fail_fulltext=None, delivery=None):
    llm = FakeLLM(scores, fail_screen)
    return DailyPaperPipeline(
        source=FakeSource(papers),
        keyword_filter=KeywordFilter(["video reasoning"]),
        screener=PaperScreener(llm, ["video reasoning"]),
        selector=DeepReadSelector(top_k=1, min_relevance=7),
        fulltext_reader=FakeFulltext(fail_fulltext),
        deep_summarizer=FakeDeep(),
        topics=["video reasoning"],
        delivery=delivery,
        memory_root=tmp_path,
    )


def test_pipeline_stops_when_no_hf_papers(tmp_path):
    result = build_pipeline(tmp_path, []).run(date="2026-09-03")

    assert result.counts["hf_total"] == 0
    assert result.digest_path is None
    assert result.run_path.exists()


def test_pipeline_keyword_to_screening(tmp_path):
    papers = [make_paper(1), make_paper(2, "database indexing")]

    result = build_pipeline(tmp_path, papers).run(date="2026-09-03", no_deep_read=True)

    assert result.counts["hf_total"] == 2
    assert result.counts["keyword_matched"] == 1
    assert result.counts["llm_screened"] == 1
    assert result.counts["accepted"] == 1


def test_pipeline_only_deep_reads_top_k(tmp_path):
    papers = [make_paper(1), make_paper(2), make_paper(3)]
    pipeline = build_pipeline(tmp_path, papers)

    result = pipeline.run(date="2026-09-03")

    assert result.counts["deep_read"] == 1
    assert len(pipeline.fulltext_reader.calls) == 1


def test_pipeline_survives_one_llm_failure(tmp_path):
    papers = [make_paper(1), make_paper(2)]

    result = build_pipeline(tmp_path, papers, fail_screen="2609.00001").run(
        date="2026-09-03", no_deep_read=True
    )

    assert result.counts["llm_screened"] == 1
    assert result.counts["accepted"] == 1
    assert any("screening 2609.00001" in error for error in result.errors)


def test_pipeline_survives_fulltext_failure(tmp_path):
    target = make_paper(1)

    result = build_pipeline(tmp_path, [target], fail_fulltext=[target.arxiv_id]).run(
        date="2026-09-03"
    )

    assert result.digest_path.exists()
    assert target.deep_summary_status == "fulltext_unavailable"
    assert result.counts["deep_read_success"] == 0


def test_pipeline_creates_digest(tmp_path):
    result = build_pipeline(tmp_path, [make_paper(1)]).run(date="2026-09-03")

    assert result.digest_path == tmp_path / "digests" / "2026-09-03.md"
    assert "Paper 1" in result.digest_path.read_text(encoding="utf-8")
    assert result.run_path.exists()


def test_pipeline_optional_delivery(tmp_path):
    sender = FakeDelivery()
    pipeline = build_pipeline(tmp_path, [make_paper(1)], delivery=sender)

    pipeline.run(date="2026-09-03", no_delivery=True)
    assert sender.calls == 0
    pipeline.run(date="2026-09-03")
    assert sender.calls == 1
