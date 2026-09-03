from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from arxiv_digest.models import Paper


@dataclass
class KeywordMatchResult:
    matched: bool
    matched_topics: list[str]
    matched_keywords: list[str]
    matched_fields: list[str]


@dataclass(frozen=True)
class Topic:
    name: str
    keywords: tuple[str, ...]


class KeywordFilter:
    def __init__(
        self,
        topics: list[str | dict[str, Any]],
        *,
        exclude_keywords: list[str] | None = None,
        enabled: bool = True,
    ):
        self.topics = _parse_topics(topics)
        self.exclude_keywords = exclude_keywords or []
        self.enabled = enabled

    @classmethod
    def from_config(
        cls,
        topics_config: dict[str, Any],
        preferences: dict[str, Any] | None = None,
    ) -> KeywordFilter:
        filter_config = (preferences or topics_config).get("keyword_filter", {})
        return cls(
            topics_config.get("topics", []),
            exclude_keywords=topics_config.get("exclude_keywords", []),
            enabled=filter_config.get("enabled", True),
        )

    def match(self, paper: Paper) -> KeywordMatchResult:
        if not self.enabled:
            return KeywordMatchResult(True, [], [], [])

        fields = _paper_fields(paper)
        if any(_matches(keyword, fields) for keyword in self.exclude_keywords):
            return KeywordMatchResult(False, [], [], [])

        matched_topics = []
        matched_keywords = []
        matched_fields = []
        for topic in self.topics:
            topic_matched = False
            for keyword in topic.keywords:
                keyword_fields = _matching_fields(keyword, fields)
                if not keyword_fields:
                    continue
                topic_matched = True
                if keyword not in matched_keywords:
                    matched_keywords.append(keyword)
                for field_name in keyword_fields:
                    if field_name not in matched_fields:
                        matched_fields.append(field_name)
            if topic_matched:
                matched_topics.append(topic.name)

        return KeywordMatchResult(
            bool(matched_topics), matched_topics, matched_keywords, matched_fields
        )

    def filter(self, papers: list[Paper]) -> list[Paper]:
        accepted = []
        for paper in papers:
            result = self.match(paper)
            paper.matched_keywords = list(result.matched_keywords)
            if result.matched:
                accepted.append(paper)
        return accepted


def _parse_topics(configured: list[str | dict[str, Any]]) -> list[Topic]:
    topics = []
    for item in configured:
        if isinstance(item, str):
            topics.append(Topic(name=item, keywords=(item,)))
            continue
        name = str(item.get("name", "")).strip()
        keywords = tuple(
            str(keyword).strip()
            for keyword in item.get("keywords", [])
            if str(keyword).strip()
        )
        if name and keywords:
            topics.append(Topic(name=name, keywords=keywords))
    return topics


def _paper_fields(paper: Paper) -> dict[str, list[str]]:
    return {
        "title": [paper.title],
        "abstract": [paper.abstract],
        "hf_summary": [paper.hf_summary or ""],
        "hf_keywords": paper.hf_keywords,
    }


def _matches(keyword: str, fields: dict[str, list[str]]) -> bool:
    return bool(_matching_fields(keyword, fields))


def _matching_fields(keyword: str, fields: dict[str, list[str]]) -> list[str]:
    needle = _normalize(keyword)
    if not needle:
        return []
    return [
        field_name
        for field_name, values in fields.items()
        if any(needle in _normalize(value) for value in values)
    ]


def _normalize(value: str) -> str:
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()
