from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date as current_date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from arxiv_digest.llm.prompts import SCREENING_PROMPT_VERSION, build_screening_messages
from arxiv_digest.models import Paper


class JsonChatClient(Protocol):
    model: str

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]: ...


class ScreeningValidationError(ValueError):
    """The LLM screening result does not match the required schema."""


@dataclass(frozen=True)
class ScreeningResult:
    relevance: float
    novelty: float
    impact: float
    matched_topic: str
    relevance_reason: str
    abstract_zh: str
    tldr_zh: str
    worth_deep_reading: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScreeningResult:
        scores = {
            name: _score(data.get(name), name)
            for name in ("relevance", "novelty", "impact")
        }
        matched_topic = data.get("matched_topic")
        if (
            isinstance(matched_topic, list)
            and matched_topic
            and all(isinstance(item, str) for item in matched_topic)
        ):
            matched_topic = matched_topic[0]
        text_fields = {
            "matched_topic": _text(matched_topic, "matched_topic"),
            **{
                name: _text(data.get(name), name)
                for name in ("relevance_reason", "abstract_zh", "tldr_zh")
            },
        }
        worth_deep_reading = data.get("worth_deep_reading")
        if (
            isinstance(worth_deep_reading, (int, float))
            and not isinstance(worth_deep_reading, bool)
            and 1 <= worth_deep_reading <= 10
        ):
            worth_deep_reading = scores["relevance"] >= 7
        elif not isinstance(worth_deep_reading, bool):
            raise ScreeningValidationError("worth_deep_reading must be a boolean")
        return cls(
            **scores,
            **text_fields,
            worth_deep_reading=worth_deep_reading,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaperScreener:
    def __init__(
        self,
        llm: JsonChatClient,
        topics: list[str | dict[str, Any]],
        *,
        cache: ScreeningCache | None = None,
        run_date: str | None = None,
        prompt_version: str = SCREENING_PROMPT_VERSION,
    ):
        self.llm = llm
        self.topics = topics
        self.cache = cache
        self.run_date = run_date or current_date.today().isoformat()
        self.prompt_version = prompt_version
        self.api_calls = 0
        self.cache_hits = 0

    def screen(self, paper: Paper, *, force: bool = False) -> ScreeningResult:
        input_hash = _input_hash(paper, self.topics)
        if self.cache and not force:
            cached = self.cache.load(
                self.run_date,
                paper.arxiv_id,
                input_hash=input_hash,
                model=self.llm.model,
                prompt_version=self.prompt_version,
            )
            if cached:
                self.cache_hits += 1
                _apply_result(paper, cached)
                return cached

        messages = build_screening_messages(paper, self.topics)
        self.api_calls += 1
        result = ScreeningResult.from_dict(self.llm.chat_json(messages))
        _apply_result(paper, result)
        if self.cache:
            self.cache.save(
                self.run_date,
                paper.arxiv_id,
                input_hash=input_hash,
                model=self.llm.model,
                prompt_version=self.prompt_version,
                result=result,
            )
        return result


class ScreeningCache:
    def __init__(self, root: Path):
        self.root = root

    def load(
        self,
        run_date: str,
        paper_id: str,
        *,
        input_hash: str,
        model: str,
        prompt_version: str,
    ) -> ScreeningResult | None:
        path = self.path_for(run_date, paper_id)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                record.get("input_hash") != input_hash
                or record.get("model") != model
                or record.get("prompt_version") != prompt_version
            ):
                return None
            return ScreeningResult.from_dict(record["result"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def save(
        self,
        run_date: str,
        paper_id: str,
        *,
        input_hash: str,
        model: str,
        prompt_version: str,
        result: ScreeningResult,
    ) -> None:
        path = self.path_for(run_date, paper_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "paper_id": paper_id,
            "input_hash": input_hash,
            "model": model,
            "prompt_version": prompt_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "result": result.to_dict(),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def path_for(self, run_date: str, paper_id: str) -> Path:
        safe_id = paper_id.replace("/", "_")
        return self.root / run_date / f"{safe_id}.json"


def filter_by_relevance(papers: list[Paper], threshold: float) -> list[Paper]:
    return [
        paper
        for paper in papers
        if paper.screening
        and float(paper.screening.get("relevance", 0)) >= threshold
    ]


def _apply_result(paper: Paper, result: ScreeningResult) -> None:
    paper.screening = result.to_dict()
    paper.abstract_zh = result.abstract_zh
    paper.tldr_zh = result.tldr_zh


def _input_hash(paper: Paper, topics: list[str | dict[str, Any]]) -> str:
    content = {
        "topics": topics,
        "paper": {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "hf_summary": paper.hf_summary,
            "hf_keywords": paper.hf_keywords,
            "hf_upvotes": paper.hf_upvotes,
        },
    }
    serialized = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _score(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScreeningValidationError(f"{field_name} must be a number from 1 to 10")
    score = float(value)
    if not 1 <= score <= 10:
        raise ScreeningValidationError(f"{field_name} must be between 1 and 10")
    return score


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ScreeningValidationError(f"{field_name} must be text")
    return value
