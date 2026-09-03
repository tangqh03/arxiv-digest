import json
from pathlib import Path

from arxiv_digest.models import Paper
from arxiv_digest.rendering.markdown import DigestStats, render_digest


DEEP_SUMMARY = json.loads(
    (
        Path(__file__).parent
        / "fixtures"
        / "llm_responses"
        / "deep_summary_valid.json"
    ).read_text(encoding="utf-8")
)


def make_paper(arxiv_id, title, relevance, *, deep=False):
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract="Original abstract.",
        abstract_zh=f"{title} 的忠实中文摘要。",
        tldr_zh=(
            f"问题：{title} 解决代理执行问题。\n"
            f"核心点：{title} 提出结构化方法。\n"
            f"意义：{title} 提升代理可靠性。"
        ),
        matched_keywords=["video reasoning"],
        screening={
            "relevance": relevance,
            "novelty": 8,
            "impact": 7,
            "relevance_reason": f"{title} 与关注主题直接相关。",
        },
        hf_url=f"https://huggingface.co/papers/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        github_url=f"https://github.com/example/{arxiv_id}",
        deep_summary=DEEP_SUMMARY if deep else None,
    )


def render(deep=None, normal=None, rejected=None):
    deep = deep or []
    normal = normal or []
    return render_digest(
        date="2026-09-03",
        topics=[{"name": "Video Reasoning", "keywords": ["video reasoning"]}],
        deep_papers=deep,
        normal_papers=normal,
        rejected_papers=rejected or [],
        stats=DigestStats(
            hf_total=8,
            keyword_matched=6,
            llm_accepted=len(deep) + len(normal),
            deep_read=len(deep),
            keyword_rejected=2,
            llm_rejected=1,
            fulltext_failed=1,
        ),
    )


def test_digest_contains_overview():
    digest = render()

    assert "# HF Daily Research Digest — 2026-09-03" in digest
    assert "HF Daily Papers: 8" in digest
    assert "关键词命中: 6" in digest
    assert "Video Reasoning" in digest


def test_digest_orders_by_relevance():
    low = make_paper("2609.00001", "Lower", 7)
    high = make_paper("2609.00002", "Higher", 9)

    digest = render(normal=[low, high])

    assert digest.index("[Higher]") < digest.index("[Lower]")


def test_digest_has_chinese_abstract():
    digest = render(normal=[make_paper("2609.00001", "中文论文", 7)])

    assert "**Abstract**:" in digest
    assert "中文论文 的忠实中文摘要。" in digest


def test_digest_has_tldr():
    digest = render(normal=[make_paper("2609.00001", "TLDR Paper", 7)])

    assert "**TL;DR**:" in digest
    assert "- 问题：TLDR Paper 解决代理执行问题。" in digest
    assert "- 核心点：TLDR Paper 提出结构化方法。" in digest
    assert "- 意义：TLDR Paper 提升代理可靠性。" in digest


def test_deep_papers_have_deep_summary():
    digest = render(deep=[make_paper("2609.00001", "Deep Paper", 9, deep=True)])

    assert "#### 深度解读" in digest
    assert "**核心直觉**" in digest
    assert DEEP_SUMMARY["core_intuition"] in digest


def test_normal_papers_do_not_have_deep_summary():
    normal = make_paper("2609.00001", "Normal Paper", 7, deep=True)
    digest = render(normal=[normal])
    section = digest[digest.index("[Normal Paper]") :]

    assert "#### 深度解读" not in section


def test_links_present():
    digest = render(normal=[make_paper("2609.00001", "Linked Paper", 7)])

    assert "[Linked Paper](https://arxiv.org/abs/2609.00001)" in digest
    assert "[HF](https://huggingface.co/papers/2609.00001)" in digest
    assert "[PDF](https://arxiv.org/pdf/2609.00001)" in digest
    assert "[GitHub](https://github.com/example/2609.00001)" in digest


def test_rejected_papers_not_rendered():
    rejected = make_paper("2609.00003", "Rejected Secret Paper", 2)

    digest = render(rejected=[rejected])

    assert "Rejected Secret Paper" not in digest


def test_digest_unicode():
    digest = render(normal=[make_paper("2609.00001", "时序推理研究 🚀", 8)])

    digest.encode("utf-8")
    assert "时序推理研究 🚀" in digest


def test_digest_no_duplicate_papers():
    duplicate = make_paper("2609.00001", "Only Once", 9, deep=True)

    digest = render(deep=[duplicate, duplicate], normal=[duplicate])

    assert digest.count("[Only Once](https://arxiv.org/abs/2609.00001)") == 1
    assert digest.count("<!-- paper -->") == 1
