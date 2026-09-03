#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build_parser():
    parser = argparse.ArgumentParser(description="Run the personalized HF Daily Papers digest")
    parser.add_argument("--date", default="today", help="today or YYYY-MM-DD")
    parser.add_argument("--max-papers", type=int)
    parser.add_argument("--deep-top-k", type=int)
    parser.add_argument("--no-deep-read", action="store_true")
    parser.add_argument("--no-delivery", action="store_true")
    parser.add_argument("--deliver", choices=["feishu"])
    parser.add_argument("--force-screening", action="store_true")
    parser.add_argument("--force-deep-summary", action="store_true")
    parser.add_argument("--force-delivery", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    from arxiv_digest.config import load_project_config
    from arxiv_digest.delivery.feishu import FeishuDelivery
    from arxiv_digest.filtering import KeywordFilter
    from arxiv_digest.fulltext.reader import FullTextReader
    from arxiv_digest.llm.client import LLMClient
    from arxiv_digest.llm.deep_summary import DeepSummarizer, DeepSummaryCache
    from arxiv_digest.llm.screening import PaperScreener, ScreeningCache
    from arxiv_digest.pipeline import DailyPaperPipeline
    from arxiv_digest.selection import DeepReadSelector
    from arxiv_digest.sources.huggingface import HuggingFacePaperSource

    topics_config, preferences = load_project_config(ROOT)
    topics = topics_config.get("topics", [])
    screening_llm = LLMClient.from_env(
        preferences.get("llm", {}),
        env_prefix="SCREENING_LLM",
        fallback_prefix="LLM",
    )
    deep_llm = LLMClient.from_env(
        preferences.get("llm", {}),
        env_prefix="DEEP_LLM",
        fallback_prefix="LLM",
    )
    delivery = None
    feishu_config = preferences.get("delivery", {}).get("feishu", {})
    wants_delivery = args.deliver == "feishu" or feishu_config.get("enabled", False)
    if wants_delivery and not args.no_delivery:
        if args.dry_run:
            import os
            webhook = os.environ.get("FEISHU_WEBHOOK_URL") or "https://dry-run.invalid"
            delivery = FeishuDelivery(
                webhook_url=webhook,
                secret=os.environ.get("FEISHU_WEBHOOK_SECRET") or None,
                max_papers_per_message=feishu_config.get("max_papers_per_message", 3),
            )
        else:
            delivery = FeishuDelivery.from_env(feishu_config)

    memory = ROOT / "memory"
    deep_config = preferences.get("deep_read", {})
    screening_config = preferences.get("screening", {})
    pipeline = DailyPaperPipeline(
        source=HuggingFacePaperSource(),
        keyword_filter=KeywordFilter.from_config(topics_config, preferences),
        screener=PaperScreener(
            screening_llm,
            topics,
            cache=ScreeningCache(memory / "analysis"),
            run_date=args.date if args.date != "today" else None,
        ),
        selector=DeepReadSelector.from_config(deep_config),
        fulltext_reader=FullTextReader(cache_root=memory / "fulltext"),
        deep_summarizer=DeepSummarizer(
            deep_llm,
            cache=DeepSummaryCache(memory / "deep_summaries"),
            max_fulltext_chars=deep_config.get("max_fulltext_chars", 120000),
            chunk_chars=deep_config.get("chunk_chars", 30000),
            chunk_overlap=deep_config.get("chunk_overlap", 1500),
        ),
        topics=topics,
        relevance_threshold=screening_config.get("relevance_threshold", 6),
        daily_limit=preferences.get("huggingface", {}).get("daily_limit", 50),
        max_screening_candidates=screening_config.get("max_candidates", 20),
        delivery=delivery,
        memory_root=memory,
    )
    try:
        result = pipeline.run(
            date=args.date,
            max_papers=args.max_papers,
            deep_top_k=args.deep_top_k,
            no_deep_read=args.no_deep_read,
            no_delivery=args.no_delivery,
            force_screening=args.force_screening,
            force_deep_summary=args.force_deep_summary,
            force_delivery=args.force_delivery,
            dry_run=args.dry_run,
        )
    finally:
        screening_llm.close()
        deep_llm.close()
        if delivery:
            delivery.close()
    for name, value in result.counts.items():
        print(f"[{name.upper()}] {value}")
    if result.digest_path:
        print(f"[DIGEST] saved {result.digest_path}")
    print(f"[RUN] saved {result.run_path}")
    if result.errors:
        print(f"[ERRORS] {len(result.errors)} recoverable errors")


if __name__ == "__main__":
    main()
