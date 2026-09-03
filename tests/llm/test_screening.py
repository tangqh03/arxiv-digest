import json
from pathlib import Path

import pytest

from arxiv_digest.llm.screening import (
    PaperScreener,
    ScreeningValidationError,
    filter_by_relevance,
)
from arxiv_digest.models import Paper


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "llm_responses"
    / "screening_valid.json"
)
TOPICS = [
    {
        "name": "Video Reasoning",
        "keywords": ["video reasoning", "temporal reasoning"],
    }
]


class FakeLLM:
    model = "fake-model"

    def __init__(self, result=None):
        self.result = result or json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.messages = []

    def chat_json(self, messages):
        self.messages.append(messages)
        return self.result


def make_paper(arxiv_id="2609.00001"):
    return Paper(
        arxiv_id=arxiv_id,
        title="VideoReasoner: Temporal Reasoning over Long Videos",
        abstract="We introduce VideoReasoner for temporal reasoning over long videos.",
        hf_summary="A hierarchical video reasoning method.",
        hf_keywords=["video reasoning"],
    )


def test_screening_request_contains_topics():
    llm = FakeLLM()

    PaperScreener(llm, TOPICS).screen(make_paper())

    user_prompt = llm.messages[0][1]["content"]
    assert "Video Reasoning" in user_prompt
    assert "temporal reasoning" in user_prompt


def test_screening_request_contains_abstract():
    llm = FakeLLM()
    paper = make_paper()

    PaperScreener(llm, TOPICS).screen(paper)

    assert paper.abstract in llm.messages[0][1]["content"]


def test_screening_prompt_requires_chinese_output():
    llm = FakeLLM()

    PaperScreener(llm, TOPICS).screen(make_paper())

    system_prompt = llm.messages[0][0]["content"]
    assert "relevance_reason、abstract_zh、tldr_zh 必须使用中文" in system_prompt


def test_screening_parses_valid_result():
    result = PaperScreener(FakeLLM(), TOPICS).screen(make_paper())

    assert result.relevance == 9
    assert result.novelty == 7
    assert result.impact == 8
    assert result.worth_deep_reading is True


def test_screening_rejects_invalid_score():
    invalid = json.loads(FIXTURE.read_text(encoding="utf-8"))
    invalid["relevance"] = 11

    with pytest.raises(ScreeningValidationError, match="between 1 and 10"):
        PaperScreener(FakeLLM(invalid), TOPICS).screen(make_paper())


def test_screening_normalizes_numeric_deep_read_hint():
    provider_result = json.loads(FIXTURE.read_text(encoding="utf-8"))
    provider_result["worth_deep_reading"] = 5

    result = PaperScreener(FakeLLM(provider_result), TOPICS).screen(make_paper())

    assert result.relevance == 9
    assert result.worth_deep_reading is True


def test_screening_normalizes_topic_list_from_provider():
    provider_result = json.loads(FIXTURE.read_text(encoding="utf-8"))
    provider_result["matched_topic"] = ["Video Reasoning", "Multimodal Agents"]

    result = PaperScreener(FakeLLM(provider_result), TOPICS).screen(make_paper())

    assert result.matched_topic == "Video Reasoning"


def test_abstract_translation_is_saved():
    paper = make_paper()

    PaperScreener(FakeLLM(), TOPICS).screen(paper)

    assert paper.abstract_zh == "本文提出 VideoReasoner，用于长视频时间推理。"


def test_tldr_is_saved():
    paper = make_paper()

    PaperScreener(FakeLLM(), TOPICS).screen(paper)

    assert paper.tldr_zh.startswith("问题：")
    assert "\n核心点：" in paper.tldr_zh
    assert "\n意义：" in paper.tldr_zh


def test_low_relevance_filtered_out():
    paper = make_paper()
    paper.screening = {"relevance": 3}

    assert filter_by_relevance([paper], threshold=6) == []


def test_high_relevance_kept():
    paper = make_paper()
    paper.screening = {"relevance": 9}

    assert filter_by_relevance([paper], threshold=6) == [paper]
