from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from arxiv_digest.models import Paper
from arxiv_digest.rendering.markdown import DigestStats, format_tldr_bullets
from arxiv_digest.selection import rank_papers


DEFAULT_RECEIPT_ROOT = Path(__file__).resolve().parents[2] / "memory" / "delivery"


class FeishuConfigurationError(ValueError):
    """Feishu delivery configuration is incomplete."""


@dataclass(frozen=True)
class DeliveryResult:
    success: bool
    message_count: int
    error: str | None = None
    skipped: bool = False


class FeishuDelivery:
    def __init__(
        self,
        *,
        webhook_url: str,
        secret: str | None = None,
        max_papers_per_message: int = 3,
        receipt_root: Path = DEFAULT_RECEIPT_ROOT,
        transport: httpx.BaseTransport | None = None,
    ):
        if not webhook_url:
            raise FeishuConfigurationError("FEISHU_WEBHOOK_URL is required")
        self._webhook_url = webhook_url
        self._secret = secret
        self.max_papers_per_message = max(1, max_papers_per_message)
        self.receipt_root = receipt_root
        self._http = httpx.Client(timeout=30, transport=transport)
        self.last_payloads: list[dict[str, Any]] = []

    @classmethod
    def from_env(cls, config: dict[str, Any] | None = None, **kwargs):
        settings = config or {}
        return cls(
            webhook_url=os.environ.get("FEISHU_WEBHOOK_URL", ""),
            secret=os.environ.get("FEISHU_WEBHOOK_SECRET") or None,
            max_papers_per_message=settings.get("max_papers_per_message", 3),
            **kwargs,
        )

    def close(self):
        self._http.close()

    def build_payloads(
        self,
        *,
        date: str,
        papers: list[Paper],
        stats: DigestStats,
        card_title: str | None = None,
        overview_note: str | None = None,
    ) -> list[dict[str, Any]]:
        ranked = rank_papers(papers)
        overview = [
            f"**HF Daily Papers:** {stats.hf_total}",
            f"**关键词命中:** {stats.keyword_matched}",
            f"**LLM 相关:** {stats.llm_accepted}",
            f"**重点精读:** {stats.deep_read}",
            "",
            "**今日论文**",
        ]
        if overview_note:
            overview.extend(["", overview_note])
        overview.extend(
            f"{index}. [{paper.title}]({paper.abs_url or paper.hf_url}) · {_score(paper, 'relevance')}/10"
            for index, paper in enumerate(ranked, 1)
        )
        payloads = [
            _card(card_title or f"HF Daily Research Digest — {date}", "\n".join(overview))
        ]
        for start in range(0, len(ranked), self.max_papers_per_message):
            group = ranked[start : start + self.max_papers_per_message]
            content = "\n\n---\n\n".join(_paper_card_text(paper) for paper in group)
            payloads.append(_card(f"论文详情 {start + 1}-{start + len(group)}", content))
        return payloads

    def send(
        self,
        *,
        date: str,
        digest: str,
        papers: list[Paper],
        stats: DigestStats,
        force: bool = False,
        dry_run: bool = False,
        card_title: str | None = None,
        overview_note: str | None = None,
    ) -> DeliveryResult:
        digest_hash = hashlib.sha256(digest.encode("utf-8")).hexdigest()
        if not force and self._already_sent(date, digest_hash):
            return DeliveryResult(True, 0, skipped=True)

        payloads = self.build_payloads(
            date=date,
            papers=papers,
            stats=stats,
            card_title=card_title,
            overview_note=overview_note,
        )
        self.last_payloads = payloads
        return self._send_payloads(date, digest_hash, payloads, dry_run=dry_run)

    def send_markdown(
        self,
        *,
        date: str,
        digest: str,
        force: bool = False,
        dry_run: bool = False,
    ) -> DeliveryResult:
        digest_hash = hashlib.sha256(digest.encode("utf-8")).hexdigest()
        if not force and self._already_sent(date, digest_hash):
            return DeliveryResult(True, 0, skipped=True)
        payloads = self.build_markdown_payloads(date=date, digest=digest)
        self.last_payloads = payloads
        return self._send_payloads(date, digest_hash, payloads, dry_run=dry_run)

    def build_markdown_payloads(self, *, date: str, digest: str) -> list[dict[str, Any]]:
        pattern = re.compile(r"<!-- paper -->(.*?)<!-- /paper -->", re.DOTALL)
        papers = [match.strip() for match in pattern.findall(digest)]
        if papers:
            overview = pattern.sub("", digest).strip()
        else:
            sections = re.split(r"(?m)(?=^###\s+)", digest)
            overview = sections[0].strip()
            papers = [section.strip() for section in sections[1:] if section.strip()]
        payloads = [_card(f"HF Daily Research Digest — {date}", overview)]
        for start in range(0, len(papers), self.max_papers_per_message):
            group = papers[start : start + self.max_papers_per_message]
            payloads.append(
                _card(
                    f"论文详情 {start + 1}-{start + len(group)}",
                    "\n\n---\n\n".join(group),
                )
            )
        return payloads

    def _send_payloads(
        self,
        date: str,
        digest_hash: str,
        payloads: list[dict[str, Any]],
        *,
        dry_run: bool,
    ) -> DeliveryResult:
        if dry_run:
            return DeliveryResult(True, len(payloads), skipped=False)

        sent = 0
        error_message = None
        for payload in payloads:
            signed_payload = dict(payload)
            if self._secret:
                timestamp = str(int(time.time()))
                signed_payload.update(
                    {"timestamp": timestamp, "sign": generate_signature(timestamp, self._secret)}
                )
            try:
                response = self._http.post(self._webhook_url, json=signed_payload)
                if response.is_error:
                    error_message = f"Feishu HTTP {response.status_code}"
                    break
                body = response.json()
                code = body.get("code", body.get("StatusCode", 0))
                if code != 0:
                    error_message = f"Feishu API error code {code}"
                    break
                sent += 1
            except (httpx.RequestError, ValueError):
                error_message = "Feishu request failed"
                break

        success = error_message is None
        self._write_receipt(
            date,
            {
                "date": date,
                "digest_hash": digest_hash,
                "sent_at": datetime.now(timezone.utc).isoformat(),
                "message_count": sent,
                "success": success,
                "error": error_message,
            },
        )
        return DeliveryResult(success, sent, error_message)

    def _already_sent(self, date: str, digest_hash: str) -> bool:
        receipt = self._load_receipt(date)
        return bool(
            receipt
            and receipt.get("success") is True
            and receipt.get("digest_hash") == digest_hash
        )

    def _load_receipt(self, date: str):
        try:
            return json.loads(self._receipt_path(date).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_receipt(self, date: str, receipt: dict[str, Any]):
        path = self._receipt_path(date)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _receipt_path(self, date: str):
        return self.receipt_root / date / "feishu.json"


def generate_signature(timestamp: str, secret: str) -> str:
    string_to_sign = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(string_to_sign, digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def _card(title: str, content: str) -> dict[str, Any]:
    return {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": title}},
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": content}}
            ],
        },
    }


def _paper_card_text(paper: Paper) -> str:
    screening = paper.screening or {}
    primary_url = paper.abs_url or paper.hf_url
    links = []
    if paper.abs_url and paper.abs_url != primary_url:
        links.append(f"[arXiv]({paper.abs_url})")
    if paper.hf_url and paper.hf_url != primary_url:
        links.append(f"[HF]({paper.hf_url})")
    title = f"[{paper.title}]({primary_url})" if primary_url else paper.title
    lines = [
        title,
        f"Relevance {_score(paper, 'relevance')}/10 · Novelty {_score(paper, 'novelty')}/10 · Impact {_score(paper, 'impact')}/10",
        "",
        "**TL;DR**:",
    ]
    lines.extend(format_tldr_bullets(paper.tldr_zh))
    lines.extend(
        [
            "",
            "**Abstract**:",
            paper.abstract_zh or paper.abstract or "暂无",
            "",
            "**推荐理由**:",
            screening.get("relevance_reason") or "暂无",
        ]
    )
    if links:
        lines.extend(["", " | ".join(links)])
    if paper.deep_summary:
        lines.extend(
            [
                f"**核心直觉:** {paper.deep_summary['core_intuition']}",
                f"**方法概览:** {paper.deep_summary['method']['overview']}",
                "**主要结果:** " + "；".join(paper.deep_summary["experiments"]["main_results"]),
            ]
        )
    return "\n".join(lines)


def _score(paper: Paper, name: str):
    value = (paper.screening or {}).get(name, 0)
    return int(value) if isinstance(value, float) and value.is_integer() else value
