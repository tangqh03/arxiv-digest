import json
from pathlib import Path

from arxiv_digest.llm.screening import PaperScreener, ScreeningCache
from arxiv_digest.models import Paper


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "llm_responses"
    / "screening_valid.json"
)
TOPICS = [{"name": "Video Reasoning", "keywords": ["video reasoning"]}]


class CountingLLM:
    model = "cache-test-model"

    def __init__(self):
        self.calls = 0

    def chat_json(self, messages):
        self.calls += 1
        return json.loads(FIXTURE.read_text(encoding="utf-8"))


def make_paper(abstract="An abstract about video reasoning."):
    return Paper(
        arxiv_id="2609.00001",
        title="Video Reasoning Paper",
        abstract=abstract,
        hf_keywords=["video reasoning"],
    )


def make_screener(tmp_path, llm):
    return PaperScreener(
        llm,
        TOPICS,
        cache=ScreeningCache(tmp_path / "analysis"),
        run_date="2026-09-03",
    )


def test_screening_cache_hit_skips_llm(tmp_path):
    llm = CountingLLM()
    screener = make_screener(tmp_path, llm)

    screener.screen(make_paper())
    cached_paper = make_paper()
    screener.screen(cached_paper)

    assert llm.calls == 1
    assert screener.api_calls == 1
    assert screener.cache_hits == 1
    assert cached_paper.abstract_zh
    cache_file = tmp_path / "analysis" / "2026-09-03" / "2609.00001.json"
    record = json.loads(cache_file.read_text(encoding="utf-8"))
    assert record["model"] == "cache-test-model"
    assert record["prompt_version"] == "screening-v3"


def test_screening_cache_invalidates_when_abstract_changes(tmp_path):
    llm = CountingLLM()
    screener = make_screener(tmp_path, llm)

    screener.screen(make_paper("Original abstract."))
    screener.screen(make_paper("Changed abstract."))

    assert llm.calls == 2
    assert screener.cache_hits == 0


def test_force_screening_ignores_cache(tmp_path):
    llm = CountingLLM()
    screener = make_screener(tmp_path, llm)

    screener.screen(make_paper())
    screener.screen(make_paper(), force=True)

    assert llm.calls == 2
    assert screener.cache_hits == 0
