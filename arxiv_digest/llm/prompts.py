from __future__ import annotations

import json
from typing import Any

from arxiv_digest.models import Paper


SCREENING_PROMPT_VERSION = "screening-v3"


SCREENING_SYSTEM_PROMPT = """你是在做研究论文筛选，不是在判断论文质量本身。

relevance 表示论文与用户关注主题的直接相关程度。不要因为作者知名、benchmark 很强或模型规模很大而给高 relevance。只有论文的问题或方法与 topics 有直接联系时才给高分。

请同时完成 abstract 中文翻译和简短 TL;DR。relevance_reason、abstract_zh、tldr_zh 必须使用中文，但方法名、模型名和 benchmark 名保留原文。abstract_zh 必须忠实于原文，保留所有数字，不得加入原 abstract 中不存在的结论。tldr_zh 必须是一个包含三行的 JSON string，不要包含 Markdown 列表符号；三行依次以“问题：”“核心点：”“意义：”开头，分别说明论文解决什么、核心 idea、为什么值得看。

仅返回 JSON object，字段必须是：relevance、novelty、impact、matched_topic、relevance_reason、abstract_zh、tldr_zh、worth_deep_reading。三个分数均为 1-10。matched_topic 只能是一个最匹配 topic 的 JSON string，不能是数组。worth_deep_reading 只能是 JSON boolean true 或 false，绝不能返回数字评分。"""


def build_screening_messages(
    paper: Paper, topics: list[str | dict[str, Any]]
) -> list[dict[str, str]]:
    metadata = {
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
    return [
        {"role": "system", "content": SCREENING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(metadata, ensure_ascii=False, indent=2),
        },
    ]
