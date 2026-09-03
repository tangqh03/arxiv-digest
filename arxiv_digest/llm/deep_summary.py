from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arxiv_digest.llm.screening import JsonChatClient
from arxiv_digest.models import Paper


DEEP_SUMMARY_PROMPT_VERSION = "deep-summary-v1"

SYSTEM_PROMPT = """你正在根据论文全文生成结构化中文研究解读。不要复述 abstract；方法部分必须来自正文 pipeline。实验数字只有在提供的文本明确支持时才能写，绝不能编造 accuracy、benchmark、提升比例或模型规模。limitations 中应优先记录作者明确陈述的限制；合理推断必须明确标注为推断。仅返回符合指定 schema 的 JSON object。"""


class DeepSummaryValidationError(ValueError):
    """Deep-summary JSON does not match the required schema."""


class DeepSummaryCache:
    def __init__(self, root: Path):
        self.root = root

    def load(self, paper_id: str, *, input_hash: str, model: str, prompt_version: str):
        try:
            record = json.loads(self.path_for(paper_id).read_text(encoding="utf-8"))
            if (
                record.get("input_hash") != input_hash
                or record.get("model") != model
                or record.get("prompt_version") != prompt_version
            ):
                return None
            return validate_deep_summary(record["result"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def save(self, paper_id: str, *, input_hash: str, model: str, prompt_version: str, result):
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "paper_id": paper_id,
            "input_hash": input_hash,
            "model": model,
            "prompt_version": prompt_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
        path = self.path_for(paper_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def path_for(self, paper_id: str) -> Path:
        return self.root / f"{paper_id.replace('/', '_')}.json"


class DeepSummarizer:
    def __init__(
        self,
        llm: JsonChatClient,
        *,
        cache: DeepSummaryCache | None = None,
        max_fulltext_chars: int = 120000,
        chunk_chars: int = 30000,
        chunk_overlap: int = 1500,
        prompt_version: str = DEEP_SUMMARY_PROMPT_VERSION,
    ):
        self.llm = llm
        self.cache = cache
        self.max_fulltext_chars = max_fulltext_chars
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.prompt_version = prompt_version
        self.api_calls = 0
        self.cache_hits = 0

    def summarize(self, paper: Paper, fulltext: str, *, force: bool = False) -> dict[str, Any]:
        input_hash = hashlib.sha256(fulltext.encode("utf-8")).hexdigest()
        if self.cache and not force:
            cached = self.cache.load(
                paper.arxiv_id,
                input_hash=input_hash,
                model=self.llm.model,
                prompt_version=self.prompt_version,
            )
            if cached:
                self.cache_hits += 1
                paper.deep_summary = cached
                paper.deep_summary_status = "success"
                return cached

        if len(fulltext) <= self.max_fulltext_chars:
            final_input = {"fulltext": fulltext}
        else:
            chunks = chunk_fulltext(fulltext, self.chunk_chars, self.chunk_overlap)
            notes = [self._chunk_notes(paper, chunk, index, len(chunks)) for index, chunk in enumerate(chunks, 1)]
            final_input = {"notes": notes}

        validated = self._validated_final_summary(paper, **final_input)
        paper.deep_summary = validated
        paper.deep_summary_status = "success"
        if self.cache:
            self.cache.save(
                paper.arxiv_id,
                input_hash=input_hash,
                model=self.llm.model,
                prompt_version=self.prompt_version,
                result=validated,
            )
        return validated

    def _validated_final_summary(self, paper: Paper, **final_input):
        for attempt in range(2):
            result = self._final_summary(paper, **final_input)
            try:
                return validate_deep_summary(result)
            except DeepSummaryValidationError:
                if attempt == 1:
                    raise
        raise AssertionError("unreachable")

    def summarize_safe(self, paper: Paper, fulltext: str, *, force: bool = False):
        try:
            return self.summarize(paper, fulltext, force=force)
        except Exception:
            paper.deep_summary = None
            paper.deep_summary_status = "failed"
            return None

    def _chunk_notes(self, paper: Paper, chunk: str, index: int, total: int):
        prompt = {
            "stage": "chunk_notes",
            "paper": {"title": paper.title, "arxiv_id": paper.arxiv_id},
            "chunk": {"index": index, "total": total, "text": chunk},
            "instructions": "提取本段明确支持的问题、方法、实验、数字和限制；不要补充段外事实。",
        }
        self.api_calls += 1
        return self.llm.chat_json(_messages(prompt))

    def _final_summary(self, paper: Paper, *, fulltext: str | None = None, notes=None):
        prompt = {
            "stage": "final_synthesis",
            "paper": {"title": paper.title, "arxiv_id": paper.arxiv_id},
            "required_schema": {
                "one_sentence": "",
                "problem": {"what": "", "why_important": ""},
                "prior_work_gap": "",
                "core_intuition": "",
                "method": {"overview": "", "pipeline": []},
                "experiments": {"setup": "", "main_results": []},
                "limitations": [],
                "why_relevant_to_user": "",
                "takeaway": "",
            },
            "fulltext": fulltext,
            "chunk_notes": notes,
        }
        self.api_calls += 1
        return self.llm.chat_json(_messages(prompt))


def chunk_fulltext(text: str, chunk_chars: int, overlap: int = 0) -> list[str]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    overlap = max(0, min(overlap, chunk_chars - 1))
    matches = list(re.finditer(r"(?m)^#{1,4}\s+.+$", text))
    if not matches:
        return _length_chunks(text, chunk_chars, overlap)

    sections = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        sections.append(text[: matches[0].start()].strip())
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(text[match.start() : end].strip())

    chunks = []
    current = ""
    for section in sections:
        if len(section) > chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_length_chunks(section, chunk_chars, overlap))
        elif not current:
            current = section
        elif len(current) + 2 + len(section) <= chunk_chars:
            current = f"{current}\n\n{section}"
        else:
            chunks.append(current)
            current = section
    if current:
        chunks.append(current)
    return chunks


def validate_deep_summary(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise DeepSummaryValidationError("deep summary must be an object")
    for name in (
        "one_sentence",
        "prior_work_gap",
        "core_intuition",
        "why_relevant_to_user",
        "takeaway",
    ):
        _require_text(data, name)
    problem = _require_dict(data, "problem")
    _require_text(problem, "what")
    _require_text(problem, "why_important")
    method = _require_dict(data, "method")
    _require_text(method, "overview")
    _require_text_list(method, "pipeline")
    experiments = _require_dict(data, "experiments")
    _require_text(experiments, "setup")
    _require_text_list(experiments, "main_results")
    _require_text_list(data, "limitations")
    return data


def _messages(payload):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _length_chunks(text: str, chunk_chars: int, overlap: int) -> list[str]:
    step = chunk_chars - overlap
    return [text[start : start + chunk_chars] for start in range(0, len(text), step)]


def _require_dict(data, name):
    value = data.get(name)
    if not isinstance(value, dict):
        raise DeepSummaryValidationError(f"{name} must be an object")
    return value


def _require_text(data, name):
    if not isinstance(data.get(name), str):
        raise DeepSummaryValidationError(f"{name} must be text")


def _require_text_list(data, name):
    value = data.get(name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DeepSummaryValidationError(f"{name} must be a list of text")
