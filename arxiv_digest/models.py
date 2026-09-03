from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    published: str | None = None
    updated: str | None = None
    categories: list[str] = field(default_factory=list)
    abs_url: str | None = None
    pdf_url: str | None = None
    comment: str = ""

    sources: list[str] = field(default_factory=list)

    hf_url: str | None = None
    hf_upvotes: int | None = None
    hf_summary: str | None = None
    hf_keywords: list[str] = field(default_factory=list)
    project_url: str | None = None
    github_url: str | None = None

    matched_keywords: list[str] = field(default_factory=list)
    screening: dict[str, Any] | None = None
    abstract_zh: str | None = None
    tldr_zh: str | None = None

    fulltext_path: str | None = None
    deep_summary: dict[str, Any] | None = None
    deep_summary_status: str | None = None

    legacy_extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.sources = _unique(self.sources)

    @classmethod
    def from_legacy_dict(cls, data: dict[str, Any]) -> Paper:
        mapped_keys = {
            "arxiv_id",
            "title",
            "authors",
            "abstract",
            "summary",
            "published",
            "updated",
            "categories",
            "abs_url",
            "abs",
            "pdf_url",
            "pdf",
            "comment",
            "sources",
            "hf_url",
            "hf_upvotes",
            "hf_summary",
            "hf_keywords",
            "project_url",
            "github_url",
            "matched_keywords",
            "screening",
            "abstract_zh",
            "tldr_zh",
            "fulltext_path",
            "deep_summary",
            "deep_summary_status",
        }
        return cls(
            arxiv_id=data.get("arxiv_id", ""),
            title=data.get("title", ""),
            authors=list(data.get("authors") or []),
            abstract=data.get("abstract", data.get("summary", "")),
            published=data.get("published"),
            updated=data.get("updated"),
            categories=list(data.get("categories") or []),
            abs_url=data.get("abs_url", data.get("abs")),
            pdf_url=data.get("pdf_url", data.get("pdf")),
            comment=data.get("comment", ""),
            sources=list(data.get("sources") or []),
            hf_url=data.get("hf_url"),
            hf_upvotes=data.get("hf_upvotes"),
            hf_summary=data.get("hf_summary"),
            hf_keywords=list(data.get("hf_keywords") or []),
            project_url=data.get("project_url"),
            github_url=data.get("github_url"),
            matched_keywords=list(data.get("matched_keywords") or []),
            screening=data.get("screening"),
            abstract_zh=data.get("abstract_zh"),
            tldr_zh=data.get("tldr_zh"),
            fulltext_path=data.get("fulltext_path"),
            deep_summary=data.get("deep_summary"),
            deep_summary_status=data.get("deep_summary_status"),
            legacy_extra={key: value for key, value in data.items() if key not in mapped_keys},
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        data = dict(self.legacy_extra)
        data.update(
            {
                "arxiv_id": self.arxiv_id,
                "title": self.title,
                "summary": self.abstract,
                "published": self.published or "",
                "updated": self.updated or "",
                "comment": self.comment,
                "authors": list(self.authors),
                "categories": list(self.categories),
                "pdf": self.pdf_url,
                "abs": self.abs_url,
                "sources": list(self.sources),
            }
        )
        optional = {
            "hf_url": self.hf_url,
            "hf_upvotes": self.hf_upvotes,
            "hf_summary": self.hf_summary,
            "hf_keywords": self.hf_keywords or None,
            "project_url": self.project_url,
            "github_url": self.github_url,
            "matched_keywords": self.matched_keywords or None,
            "screening": self.screening,
            "abstract_zh": self.abstract_zh,
            "tldr_zh": self.tldr_zh,
            "fulltext_path": self.fulltext_path,
            "deep_summary": self.deep_summary,
            "deep_summary_status": self.deep_summary_status,
        }
        data.update({key: value for key, value in optional.items() if value is not None})
        return data
