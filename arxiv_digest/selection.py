from __future__ import annotations

from dataclasses import dataclass

from arxiv_digest.models import Paper


@dataclass(frozen=True)
class SelectionResult:
    normal_candidates: list[Paper]
    deep_read_candidates: list[Paper]


class DeepReadSelector:
    def __init__(
        self,
        *,
        enabled: bool = True,
        top_k: int = 3,
        min_relevance: float = 7,
    ):
        self.enabled = enabled
        self.top_k = max(0, top_k)
        self.min_relevance = min_relevance

    @classmethod
    def from_config(cls, config: dict) -> DeepReadSelector:
        return cls(
            enabled=config.get("enabled", True),
            top_k=config.get("top_k", 3),
            min_relevance=config.get("min_relevance", 7),
        )

    def select(self, papers: list[Paper]) -> SelectionResult:
        ranked = sorted(papers, key=_ranking_key)
        if not self.enabled or self.top_k == 0:
            return SelectionResult(ranked, [])

        eligible = [
            paper
            for paper in ranked
            if _screening_score(paper, "relevance") >= self.min_relevance
        ]
        deep_read = eligible[: self.top_k]
        selected = {id(paper) for paper in deep_read}
        normal = [paper for paper in ranked if id(paper) not in selected]
        return SelectionResult(normal, deep_read)


def rank_papers(papers: list[Paper]) -> list[Paper]:
    return sorted(papers, key=_ranking_key)


def _ranking_key(paper: Paper) -> tuple:
    return (
        -_screening_score(paper, "relevance"),
        -_screening_score(paper, "novelty"),
        -_screening_score(paper, "impact"),
        -(paper.hf_upvotes or 0),
        paper.arxiv_id,
        paper.title.casefold(),
    )


def _screening_score(paper: Paper, name: str) -> float:
    if not paper.screening:
        return 0
    value = paper.screening.get(name, 0)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
