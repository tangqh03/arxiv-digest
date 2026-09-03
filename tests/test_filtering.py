from arxiv_digest.filtering import KeywordFilter
from arxiv_digest.models import Paper


TOPICS = [
    {
        "name": "video reasoning",
        "keywords": ["video reasoning", "video understanding", "temporal reasoning"],
    },
    {
        "name": "multimodal agent",
        "keywords": ["multimodal agent", "vision-language agent"],
    },
]


def paper(**kwargs):
    values = {"arxiv_id": "2609.00001", "title": "Unrelated title"}
    values.update(kwargs)
    return Paper(**values)


def test_keyword_match_title():
    result = KeywordFilter(TOPICS).match(paper(title="Efficient Video Reasoning"))

    assert result.matched is True
    assert result.matched_topics == ["video reasoning"]
    assert result.matched_keywords == ["video reasoning"]
    assert result.matched_fields == ["title"]


def test_keyword_match_abstract():
    result = KeywordFilter(TOPICS).match(
        paper(abstract="We study temporal reasoning over long recordings.")
    )

    assert result.matched is True
    assert result.matched_fields == ["abstract"]


def test_keyword_match_hf_summary():
    result = KeywordFilter(TOPICS).match(
        paper(hf_summary="A new multimodal agent for desktop tasks.")
    )

    assert result.matched_topics == ["multimodal agent"]
    assert result.matched_fields == ["hf_summary"]


def test_keyword_match_keyword_metadata():
    result = KeywordFilter(TOPICS).match(
        paper(hf_keywords=["foundation models", "video understanding"])
    )

    assert result.matched_keywords == ["video understanding"]
    assert result.matched_fields == ["hf_keywords"]


def test_keyword_match_case_insensitive():
    result = KeywordFilter(TOPICS).match(paper(title="MULTIMODAL AGENT PLANNING"))

    assert result.matched is True
    assert result.matched_topics == ["multimodal agent"]


def test_hyphen_normalization():
    result = KeywordFilter(TOPICS).match(
        paper(abstract="We introduce a vision language agent architecture.")
    )

    assert result.matched is True
    assert result.matched_keywords == ["vision-language agent"]


def test_exclude_keyword_wins():
    result = KeywordFilter(TOPICS, exclude_keywords=["survey"]).match(
        paper(title="A Survey of Video Reasoning")
    )

    assert result.matched is False
    assert result.matched_topics == []
    assert result.matched_keywords == []


def test_no_match():
    result = KeywordFilter(TOPICS).match(
        paper(abstract="A compiler optimization technique for database queries.")
    )

    assert result.matched is False
    assert result.matched_fields == []


def test_filter_disabled_returns_all():
    papers = [paper(), paper(arxiv_id="2609.00002", title="Another unrelated paper")]
    topic_config = {"topics": TOPICS, "exclude_keywords": []}
    preferences = {"keyword_filter": {"enabled": False}}

    filtered = KeywordFilter.from_config(topic_config, preferences).filter(papers)

    assert filtered == papers


def test_legacy_string_topics_are_supported():
    result = KeywordFilter(["LLM agent", "RLHF"]).match(
        paper(title="Reliable LLM Agent Training")
    )

    assert result.matched_topics == ["LLM agent"]
    assert result.matched_keywords == ["LLM agent"]
