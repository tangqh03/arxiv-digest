def make_paper(arxiv_id, title, summary=""):
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "summary": summary,
        "published": "2026-09-03",
        "updated": "2026-09-03",
        "comment": "",
        "authors": ["Ada Researcher"],
        "categories": ["cs.AI"],
        "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
        "abs": f"https://arxiv.org/abs/{arxiv_id}",
    }


def test_compute_paper_features(digest_module, monkeypatch):
    monkeypatch.setattr(digest_module, "today_str", lambda: "2026-09-03")
    paper = make_paper(
        "2609.00001",
        "An LLM Agent",
        "Reinforcement learning for reliable tool use.",
    )
    paper["comment"] = "Accepted"
    paper["authors"].append("Grace Scientist")

    features = digest_module.compute_paper_features(
        paper, ["LLM agent", "reinforcement learning", "unmatched topic"]
    )

    assert features == {
        "topic_match_count": 2,
        "is_recent": True,
        "has_comment": True,
        "author_count": 2,
    }


def test_build_raw_data_deduplicates_sources(digest_module, monkeypatch):
    monkeypatch.setattr(digest_module, "today_str", lambda: "2026-09-03")
    topic = make_paper("2609.00001", "Shared Paper", "LLM agent research")
    cross = make_paper("2609.00001", "Shared Paper", "Duplicate source")
    hf = make_paper("2609.00002", "HF Paper", "Another abstract")

    raw = digest_module.build_raw_data(
        [topic], [cross], [hf], history={}, prefs={}, topics=["LLM agent"]
    )

    assert raw["total_papers"] == 2
    by_id = {paper["arxiv_id"]: paper for paper in raw["papers"]}
    assert set(by_id["2609.00001"]["sources"]) == {
        "arxiv-topic",
        "alphaxiv-trending",
    }
    assert by_id["2609.00002"]["sources"] == ["huggingface-daily"]


def test_hf_id_parser_preserves_order(digest_module, monkeypatch):
    html = """
    <a href="/papers/2609.00003">third</a>
    <a href="/papers/2609.00001">first</a>
    <a href="/papers/2609.00003">third duplicate</a>
    <a href="/papers/2609.00002">second</a>
    """
    monkeypatch.setattr(digest_module, "fetch_url", lambda *args, **kwargs: html)

    assert digest_module.fetch_huggingface_ids(max_papers=2) == [
        "2609.00003",
        "2609.00001",
        "2609.00002",
    ]


def test_digest_formatter_contains_title_and_links(digest_module):
    paper = make_paper("2609.00001", "A Testable Paper", "A concise abstract.")

    rendered = digest_module.format_paper(
        paper, history={}, prefs={"summary": {"max_length": 800}}
    )

    assert "### A Testable Paper" in rendered
    assert "[https://arxiv.org/abs/2609.00001]" in rendered
    assert "[https://arxiv.org/pdf/2609.00001]" in rendered
