from arxiv_digest.models import Paper
from arxiv_digest.selection import DeepReadSelector


def make_paper(
    arxiv_id,
    *,
    relevance,
    novelty=5,
    impact=5,
    upvotes=0,
):
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        hf_upvotes=upvotes,
        screening={
            "relevance": relevance,
            "novelty": novelty,
            "impact": impact,
        },
    )


def test_deep_selector_respects_top_k():
    papers = [
        make_paper("2609.00001", relevance=10),
        make_paper("2609.00002", relevance=9),
        make_paper("2609.00003", relevance=8),
    ]

    result = DeepReadSelector(top_k=2, min_relevance=7).select(papers)

    assert [paper.arxiv_id for paper in result.deep_read_candidates] == [
        "2609.00001",
        "2609.00002",
    ]
    assert [paper.arxiv_id for paper in result.normal_candidates] == ["2609.00003"]


def test_deep_selector_requires_min_relevance():
    papers = [
        make_paper("2609.00001", relevance=8),
        make_paper("2609.00002", relevance=6.9, novelty=10, impact=10),
    ]

    result = DeepReadSelector(top_k=3, min_relevance=7).select(papers)

    assert [paper.arxiv_id for paper in result.deep_read_candidates] == ["2609.00001"]
    assert [paper.arxiv_id for paper in result.normal_candidates] == ["2609.00002"]


def test_relevance_dominates_popularity():
    relevant = make_paper("2609.00001", relevance=9, upvotes=10)
    popular = make_paper(
        "2609.00002", relevance=3, novelty=10, impact=10, upvotes=1000
    )

    result = DeepReadSelector(top_k=1, min_relevance=1).select([popular, relevant])

    assert result.deep_read_candidates == [relevant]


def test_selector_is_deterministic():
    first = make_paper("2609.00001", relevance=8, novelty=7, impact=6, upvotes=5)
    second = make_paper("2609.00002", relevance=8, novelty=7, impact=6, upvotes=5)
    selector = DeepReadSelector(top_k=2, min_relevance=7)

    forward = selector.select([second, first]).deep_read_candidates
    reverse = selector.select([first, second]).deep_read_candidates

    assert [paper.arxiv_id for paper in forward] == ["2609.00001", "2609.00002"]
    assert [paper.arxiv_id for paper in reverse] == ["2609.00001", "2609.00002"]


def test_deep_read_disabled():
    papers = [
        make_paper("2609.00002", relevance=8),
        make_paper("2609.00001", relevance=9),
    ]

    result = DeepReadSelector(enabled=False).select(papers)

    assert result.deep_read_candidates == []
    assert [paper.arxiv_id for paper in result.normal_candidates] == [
        "2609.00001",
        "2609.00002",
    ]
