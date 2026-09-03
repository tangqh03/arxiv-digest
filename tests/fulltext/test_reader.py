from pathlib import Path

from arxiv_digest.fulltext.reader import FullTextReader
from arxiv_digest.models import Paper


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "paper_fulltext.md"


def valid_text():
    return FIXTURE.read_text(encoding="utf-8")


def make_paper():
    return Paper(
        arxiv_id="2609.00001",
        title="Test Paper on Video Reasoning",
        abstract="This paper studies reliable temporal reasoning over long videos.",
        pdf_url="https://arxiv.org/pdf/2609.00001",
    )


def fail(*args):
    raise RuntimeError("unavailable")


def test_fulltext_prefers_hf_markdown(tmp_path):
    calls = {"html": 0, "pdf": 0}

    def html(url):
        calls["html"] += 1
        return ""

    def pdf(url):
        calls["pdf"] += 1
        return b""

    result = FullTextReader(
        hf_reader=lambda paper_id: valid_text(),
        html_fetcher=html,
        pdf_downloader=pdf,
        cache_root=tmp_path,
    ).read(make_paper())

    assert result.source == "hf"
    assert result.cached is False
    assert calls == {"html": 0, "pdf": 0}


def test_fulltext_falls_back_to_arxiv_html(tmp_path):
    body = valid_text().replace("\n", "</p><p>")
    html = f"<html><nav>menu</nav><article><p>{body}</p></article><script>bad()</script></html>"
    result = FullTextReader(
        hf_reader=fail,
        html_fetcher=lambda url: html,
        pdf_downloader=fail,
        cache_root=tmp_path,
    ).read(make_paper())

    assert result.source == "arxiv-html"
    assert "menu" not in result.text
    assert "bad()" not in result.text


def test_fulltext_falls_back_to_pdf(tmp_path):
    result = FullTextReader(
        hf_reader=fail,
        html_fetcher=fail,
        pdf_downloader=lambda url: b"fake-pdf",
        pdf_extractor=lambda data: valid_text(),
        cache_root=tmp_path,
    ).read(make_paper())

    assert result.source == "arxiv-pdf"
    assert (tmp_path / "2609.00001" / "paper.pdf").read_bytes() == b"fake-pdf"


def test_fulltext_uses_cache(tmp_path):
    paper = make_paper()
    first = FullTextReader(
        hf_reader=lambda paper_id: valid_text(), cache_root=tmp_path
    ).read(paper)
    second_paper = make_paper()
    second = FullTextReader(
        hf_reader=fail,
        html_fetcher=fail,
        pdf_downloader=fail,
        cache_root=tmp_path,
    ).read(second_paper)

    assert first.cached is False
    assert second.cached is True
    assert second.source == "hf"
    assert second_paper.fulltext_path.endswith("paper.md")


def test_invalid_cached_text_is_refetched(tmp_path):
    directory = tmp_path / "2609.00001"
    directory.mkdir()
    (directory / "paper.md").write_text("garbage", encoding="utf-8")
    calls = 0

    def hf_reader(paper_id):
        nonlocal calls
        calls += 1
        return valid_text()

    result = FullTextReader(hf_reader=hf_reader, cache_root=tmp_path).read(make_paper())

    assert result.cached is False
    assert result.source == "hf"
    assert calls == 1


def test_short_garbage_text_is_rejected(tmp_path):
    result = FullTextReader(
        hf_reader=lambda paper_id: "Introduction: not a paper",
        html_fetcher=fail,
        pdf_downloader=fail,
        cache_root=tmp_path,
    ).read(make_paper())

    assert result.text == ""
    assert result.source is None
    assert "invalid or incomplete text" in result.error


def test_all_sources_failed_returns_graceful_result(tmp_path):
    result = FullTextReader(
        hf_reader=fail,
        html_fetcher=fail,
        pdf_downloader=fail,
        cache_root=tmp_path,
    ).read(make_paper())

    assert result.text == ""
    assert result.source is None
    assert result.cached is False
    assert "hf: RuntimeError" in result.error
