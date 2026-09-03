from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arxiv_digest.models import Paper
from arxiv_digest.selection import rank_papers


@dataclass(frozen=True)
class DigestStats:
    hf_total: int
    keyword_matched: int
    llm_accepted: int
    deep_read: int
    keyword_rejected: int = 0
    llm_rejected: int = 0
    fulltext_failed: int = 0


def render_digest(
    *,
    date: str,
    topics: list[str | dict[str, Any]],
    deep_papers: list[Paper],
    normal_papers: list[Paper],
    stats: DigestStats,
    rejected_papers: list[Paper] | None = None,
) -> str:
    del rejected_papers  # Rejected papers are counted but intentionally never rendered.
    deep = _deduplicate(rank_papers(deep_papers))
    deep_ids = {_identity(paper) for paper in deep}
    normal = _deduplicate(
        paper for paper in rank_papers(normal_papers) if _identity(paper) not in deep_ids
    )

    lines = [f"# HF Daily Research Digest — {date}", "", "## 今日概览", ""]
    lines.extend(
        [
            f"- HF Daily Papers: {stats.hf_total}",
            f"- 关键词命中: {stats.keyword_matched}",
            f"- LLM 相关: {stats.llm_accepted}",
            f"- 重点精读: {stats.deep_read}",
            "",
            "关注主题:",
        ]
    )
    lines.extend(f"- {_topic_name(topic)}" for topic in topics)
    lines.extend(["", "---", "", "## ⭐ 今日最值得读", ""])
    if deep:
        for paper in deep:
            lines.extend(_render_paper(paper, deep=True))
    else:
        lines.extend(["_今日没有达到精读阈值的论文。_", ""])

    lines.extend(["## 📚 其他相关论文", ""])
    if normal:
        for paper in normal:
            lines.extend(_render_paper(paper, deep=False))
    else:
        lines.extend(["_暂无其他相关论文。_", ""])

    lines.extend(
        [
            "## 🗑️ 筛选统计",
            "",
            f"- 关键词未命中: {stats.keyword_rejected}",
            f"- LLM relevance 低于阈值: {stats.llm_rejected}",
            f"- 全文读取失败: {stats.fulltext_failed}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_paper(paper: Paper, *, deep: bool) -> list[str]:
    screening = paper.screening or {}
    primary_url = paper.abs_url or paper.hf_url
    title = f"[{paper.title}]({primary_url})" if primary_url else paper.title
    lines = [
        "<!-- paper -->",
        title,
        f"Relevance {_score(screening, 'relevance')}/10 · "
        f"Novelty {_score(screening, 'novelty')}/10 · "
        f"Impact {_score(screening, 'impact')}/10",
        "",
        "**TL;DR**:",
    ]
    lines.extend(format_tldr_bullets(paper.tldr_zh))
    lines.extend(
        [
            "",
            "**Abstract**:",
            "",
            paper.abstract_zh or paper.abstract or "_暂无摘要。_",
            "",
            "**推荐理由**:",
            "",
            screening.get("relevance_reason") or "_暂无推荐说明。_",
            "",
        ]
    )
    links = _links(paper, primary_url)
    if links:
        lines.extend([" · ".join(links), ""])
    if deep and paper.deep_summary:
        lines.extend(_render_deep_summary(paper.deep_summary))
    lines.extend(["<!-- /paper -->", "", "---", ""])
    return lines


def _render_deep_summary(summary: dict[str, Any]) -> list[str]:
    problem = summary["problem"]
    method = summary["method"]
    experiments = summary["experiments"]
    lines = [
        "#### 深度解读",
        "",
        "**研究问题**",
        "",
        f"{problem['what']} {problem['why_important']}",
        "",
        "**已有工作缺口**",
        "",
        summary["prior_work_gap"],
        "",
        "**核心直觉**",
        "",
        summary["core_intuition"],
        "",
        "**方法**",
        "",
        method["overview"],
        "",
    ]
    lines.extend(f"{index}. {step}" for index, step in enumerate(method["pipeline"], 1))
    lines.extend(["", "**实验**", "", experiments["setup"], ""])
    lines.extend(f"- {result}" for result in experiments["main_results"])
    lines.extend(["", "**局限**", ""])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    lines.extend(
        [
            "",
            "**与你关注方向的关系**",
            "",
            summary["why_relevant_to_user"],
            "",
            "**结论**",
            "",
            summary["takeaway"],
            "",
        ]
    )
    return lines


def format_tldr_bullets(tldr: str | None) -> list[str]:
    if not tldr:
        return ["- 暂无 TL;DR。"]
    lines = [line.strip().removeprefix("-").strip() for line in tldr.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) == 1 and not any(
        lines[0].startswith(prefix) for prefix in ("问题：", "核心点：", "意义：")
    ):
        lines[0] = f"核心点：{lines[0]}"
    return [f"- {line}" for line in lines]


def _links(paper: Paper, primary_url: str | None) -> list[str]:
    candidates = (
        ("HF", paper.hf_url),
        ("arXiv", paper.abs_url),
        ("PDF", paper.pdf_url),
        ("GitHub", paper.github_url),
        ("Project", paper.project_url),
    )
    return [
        f"[{label}]({url})"
        for label, url in candidates
        if url and url != primary_url
    ]


def _score(screening: dict[str, Any], name: str):
    value = screening.get(name, 0)
    return int(value) if isinstance(value, float) and value.is_integer() else value


def _identity(paper: Paper) -> str:
    return paper.arxiv_id or paper.title


def _deduplicate(papers) -> list[Paper]:
    seen = set()
    result = []
    for paper in papers:
        identity = _identity(paper)
        if identity not in seen:
            seen.add(identity)
            result.append(paper)
    return result


def _topic_name(topic: str | dict[str, Any]) -> str:
    return topic if isinstance(topic, str) else str(topic.get("name", ""))
