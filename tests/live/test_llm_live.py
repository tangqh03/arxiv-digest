import os

import pytest

from arxiv_digest.llm.client import LLMClient
from arxiv_digest.llm.deep_summary import DeepSummarizer
from arxiv_digest.llm.screening import PaperScreener
from arxiv_digest.models import Paper


pytestmark = pytest.mark.live


def test_llm_screening_live():
    required = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
    if not all(os.environ.get(name) for name in required):
        pytest.skip("LLM live-test credentials are not configured")

    paper = Paper(
        arxiv_id="live-test",
        title="VideoReasoner: Structured Temporal Reasoning over Long Videos",
        abstract=(
            "We study temporal reasoning over long videos and propose a structured "
            "pipeline that retrieves relevant events before answering questions."
        ),
        hf_keywords=["video reasoning", "temporal reasoning"],
    )
    with LLMClient.from_env({"max_retries": 1, "timeout_seconds": 60}) as llm:
        result = PaperScreener(llm, ["video reasoning"]).screen(paper)

    assert 1 <= result.relevance <= 10
    assert result.abstract_zh
    assert result.tldr_zh


def test_deep_llm_live():
    required = ("DEEP_LLM_API_KEY", "DEEP_LLM_BASE_URL", "DEEP_LLM_MODEL")
    if not all(os.environ.get(name) for name in required):
        pytest.skip("Deep LLM live-test credentials are not configured")

    paper = Paper(
        arxiv_id="deep-live-test",
        title="Structured Tool Use for Reliable Agents",
    )
    fulltext = """# Structured Tool Use for Reliable Agents

## Abstract
We study reliable tool use for language-model agents.

## Introduction
Agents can select invalid tools or ignore failed calls. Reliable execution matters because these errors compound across long tasks.

## Method
Our pipeline first plans a tool call, validates its arguments against a schema, executes it, and verifies the returned observation before continuing. Failed validation triggers replanning.

## Experiments
We compare the complete pipeline with variants that remove argument validation or observation verification. The complete system makes fewer unsupported decisions. Removing verification causes errors to propagate across later steps.

## Limitations
The authors evaluate only text-based tools and do not test embodied agents.

## Conclusion
Validation and observation verification improve the reliability of multi-step agent execution.
"""
    with LLMClient.from_env(
        {"max_retries": 1, "timeout_seconds": 120}, env_prefix="DEEP_LLM"
    ) as llm:
        result = DeepSummarizer(llm).summarize(paper, fulltext)

    assert result["problem"]["what"]
    assert result["method"]["pipeline"]
    assert result["experiments"]["main_results"]
    assert result["limitations"]
