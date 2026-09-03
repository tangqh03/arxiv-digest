import json
from pathlib import Path

import pytest

from arxiv_digest.llm.deep_summary import (
    DeepSummarizer,
    DeepSummaryCache,
    DeepSummaryValidationError,
    chunk_fulltext,
)
from arxiv_digest.models import Paper


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def valid_result():
    return json.loads(
        (FIXTURES / "llm_responses" / "deep_summary_valid.json").read_text(
            encoding="utf-8"
        )
    )


def fulltext():
    return (FIXTURES / "paper_fulltext.md").read_text(encoding="utf-8")


def paper():
    return Paper(arxiv_id="2609.00001", title="Test Paper on Video Reasoning")


class FakeLLM:
    model = "deep-test-model"

    def __init__(self, final=None, error=None):
        self.final = valid_result() if final is None else final
        self.error = error
        self.calls = []

    def chat_json(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        payload = json.loads(messages[1]["content"])
        if payload["stage"] == "chunk_notes":
            return {"method": "chunk evidence", "limitations": []}
        return self.final


def test_deep_summary_uses_fulltext():
    llm = FakeLLM()
    DeepSummarizer(llm).summarize(paper(), fulltext())

    payload = json.loads(llm.calls[0][1]["content"])
    assert "explicit temporal organization" in payload["fulltext"]


def test_deep_summary_contains_required_fields():
    result = DeepSummarizer(FakeLLM()).summarize(paper(), fulltext())

    assert result["problem"]["what"]
    assert result["method"]["pipeline"]
    assert result["experiments"]["main_results"]
    assert result["limitations"]


def test_deep_summary_rejects_invalid_schema():
    invalid = valid_result()
    del invalid["method"]["pipeline"]

    with pytest.raises(DeepSummaryValidationError, match="pipeline"):
        DeepSummarizer(FakeLLM(final=invalid)).summarize(paper(), fulltext())


def test_deep_summary_retries_final_schema_without_repeating_chunks():
    class FlakyFinalLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.final_calls = 0

        def chat_json(self, messages):
            payload = json.loads(messages[1]["content"])
            self.calls.append(messages)
            if payload["stage"] == "chunk_notes":
                return {"method": "chunk evidence", "limitations": []}
            self.final_calls += 1
            if self.final_calls == 1:
                invalid = valid_result()
                del invalid["method"]["pipeline"]
                return invalid
            return valid_result()

    llm = FlakyFinalLLM()
    summarizer = DeepSummarizer(
        llm, max_fulltext_chars=500, chunk_chars=800, chunk_overlap=50
    )

    result = summarizer.summarize(paper(), fulltext())

    stages = [json.loads(call[1]["content"])["stage"] for call in llm.calls]
    assert result["method"]["pipeline"]
    assert stages.count("chunk_notes") > 1
    assert stages.count("final_synthesis") == 2


def test_short_fulltext_single_pass():
    llm = FakeLLM()
    DeepSummarizer(llm, max_fulltext_chars=10000).summarize(paper(), fulltext())

    assert len(llm.calls) == 1


def test_long_fulltext_chunked():
    llm = FakeLLM()
    DeepSummarizer(
        llm, max_fulltext_chars=500, chunk_chars=800, chunk_overlap=50
    ).summarize(paper(), fulltext())

    stages = [json.loads(call[1]["content"])["stage"] for call in llm.calls]
    assert stages.count("chunk_notes") > 1
    assert stages[-1] == "final_synthesis"


def test_section_aware_chunking():
    text = "# Paper\nintro\n\n## Method\n" + "method evidence " * 8 + "\n\n## Experiments\nresults"

    chunks = chunk_fulltext(text, chunk_chars=145, overlap=10)

    assert any(chunk.startswith("## Method") for chunk in chunks)
    assert any("## Experiments" in chunk for chunk in chunks)


def test_deep_summary_cache_hit(tmp_path):
    llm = FakeLLM()
    summarizer = DeepSummarizer(llm, cache=DeepSummaryCache(tmp_path))

    summarizer.summarize(paper(), fulltext())
    cached_paper = paper()
    summarizer.summarize(cached_paper, fulltext())

    assert len(llm.calls) == 1
    assert summarizer.cache_hits == 1
    assert cached_paper.deep_summary_status == "success"


def test_fulltext_change_invalidates_summary(tmp_path):
    llm = FakeLLM()
    summarizer = DeepSummarizer(llm, cache=DeepSummaryCache(tmp_path))

    summarizer.summarize(paper(), fulltext())
    summarizer.summarize(paper(), fulltext() + "\nNew appendix evidence.")

    assert len(llm.calls) == 2
    assert summarizer.cache_hits == 0


def test_deep_summary_failure_does_not_abort_digest():
    target = paper()
    summarizer = DeepSummarizer(FakeLLM(error=RuntimeError("provider failed")))

    result = summarizer.summarize_safe(target, fulltext())

    assert result is None
    assert target.deep_summary is None
    assert target.deep_summary_status == "failed"
