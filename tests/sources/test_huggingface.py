import json
from pathlib import Path

from arxiv_digest.sources.arxiv import ArxivPaperSource
from arxiv_digest.sources.huggingface import HuggingFacePaperSource


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class FakeApi:
    def __init__(self, records=None, error=None):
        self.records = records or []
        self.error = error
        self.calls = []

    def list_daily_papers(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return iter(self.records)


def fixture_records():
    return json.loads((FIXTURES / "hf_daily.json").read_text(encoding="utf-8"))


def make_source(records, **kwargs):
    return HuggingFacePaperSource(
        api=FakeApi(records),
        arxiv_enricher=lambda ids: [],
        **kwargs,
    )


def test_hf_daily_limit_config_preserves_legacy_limit():
    config_path = FIXTURES.parents[1] / "config" / "preferences.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["huggingface"]["daily_limit"] == 50
    assert config["huggingface"]["max_papers"] == 5


def test_hf_daily_normal_response():
    source = make_source(fixture_records())

    papers = source.list_daily("2026-09-03", limit=50)

    assert len(papers) == 3
    assert papers[0].title == "Third Paper"
    assert papers[0].hf_upvotes == 30
    assert papers[0].hf_keywords == ["agents", "reasoning"]
    assert papers[0].github_url == "https://github.com/example/third"


def test_hf_daily_keeps_original_order():
    papers = make_source(fixture_records()).list_daily("2026-09-03", limit=50)

    assert [paper.arxiv_id for paper in papers] == [
        "2609.00003",
        "2609.00001",
        "2609.00002",
    ]


def test_hf_daily_deduplicates_ids():
    records = fixture_records()
    records.insert(2, dict(records[0], title="Duplicate"))

    papers = make_source(records).list_daily("2026-09-03", limit=50)

    assert [paper.arxiv_id for paper in papers] == [
        "2609.00003",
        "2609.00001",
        "2609.00002",
    ]


def test_hf_daily_missing_optional_fields():
    paper = make_source([{"id": "2609.00004", "title": "Minimal"}]).list_daily(
        "2026-09-03", limit=50
    )[0]

    assert paper.title == "Minimal"
    assert paper.authors == []
    assert paper.hf_upvotes is None
    assert paper.hf_keywords == []


def test_hf_daily_empty_response():
    source = make_source([])

    assert source.list_daily("2026-09-03", limit=50) == []


def test_hf_primary_failure_uses_fallback():
    html = """
    <a href="/papers/2609.00002">second</a>
    <a href="/papers/2609.00001">first</a>
    <a href="/papers/2609.00002">duplicate</a>
    """
    source = HuggingFacePaperSource(
        api=FakeApi(error=RuntimeError("API unavailable")),
        cli_runner=lambda date, limit: (_ for _ in ()).throw(FileNotFoundError()),
        html_fetcher=lambda url: html,
        arxiv_enricher=lambda ids: [],
    )

    papers = source.list_daily("2026-09-03", limit=50)

    assert [paper.arxiv_id for paper in papers] == ["2609.00002", "2609.00001"]


def test_hf_arxiv_enrichment_preserves_hf_metadata():
    hf_record = fixture_records()[0]
    xml = (FIXTURES / "arxiv_feed.xml").read_text(encoding="utf-8")
    arxiv_source = ArxivPaperSource(text_fetcher=lambda url: xml)
    source = HuggingFacePaperSource(
        api=FakeApi([hf_record]), arxiv_enricher=arxiv_source.fetch_by_ids
    )

    paper = source.list_daily("2026-09-03", limit=50)[0]

    assert paper.title == "Third Paper"
    assert paper.abstract == "Full arXiv abstract for the third paper."
    assert paper.authors == ["ArXiv Author"]
    assert paper.categories == ["cs.AI"]
    assert paper.hf_summary == "HF summary for the third paper."
    assert paper.hf_upvotes == 30
    assert paper.project_url == "https://example.com/third"
