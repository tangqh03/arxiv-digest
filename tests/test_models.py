from arxiv_digest.models import Paper


def test_paper_from_legacy_dict():
    paper = Paper.from_legacy_dict(
        {
            "arxiv_id": "2609.00001",
            "title": "A Legacy Paper",
            "summary": "Legacy abstract text.",
            "authors": ["Ada Researcher"],
            "categories": ["cs.AI"],
            "published": "2026-09-03",
            "updated": "2026-09-03",
            "abs": "https://arxiv.org/abs/2609.00001",
            "pdf": "https://arxiv.org/pdf/2609.00001",
            "comment": "Accepted",
            "sources": ["huggingface-daily"],
        }
    )

    assert paper.abstract == "Legacy abstract text."
    assert paper.abs_url == "https://arxiv.org/abs/2609.00001"
    assert paper.pdf_url == "https://arxiv.org/pdf/2609.00001"
    assert paper.sources == ["huggingface-daily"]


def test_paper_to_legacy_dict():
    paper = Paper(
        arxiv_id="2609.00001",
        title="A Normalized Paper",
        abstract="Normalized abstract text.",
        authors=["Ada Researcher"],
        categories=["cs.AI"],
        abs_url="https://arxiv.org/abs/2609.00001",
        pdf_url="https://arxiv.org/pdf/2609.00001",
        sources=["huggingface-daily"],
        abstract_zh="中文摘要",
        legacy_extra={"features": {"topic_match_count": 1}},
    )

    legacy = paper.to_legacy_dict()

    assert legacy["summary"] == "Normalized abstract text."
    assert legacy["abs"] == "https://arxiv.org/abs/2609.00001"
    assert legacy["pdf"] == "https://arxiv.org/pdf/2609.00001"
    assert legacy["abstract_zh"] == "中文摘要"
    assert legacy["features"] == {"topic_match_count": 1}


def test_missing_optional_metadata_is_allowed():
    paper = Paper.from_legacy_dict({"arxiv_id": "2609.00001", "title": "Minimal"})

    assert paper.authors == []
    assert paper.abstract == ""
    assert paper.hf_upvotes is None
    assert paper.screening is None
    assert paper.deep_summary is None


def test_sources_are_deduplicated():
    paper = Paper(
        arxiv_id="2609.00001",
        title="Deduplicated Sources",
        sources=["huggingface-daily", "arxiv", "huggingface-daily", "arxiv"],
    )

    assert paper.sources == ["huggingface-daily", "arxiv"]
