#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arxiv_digest.config import load_project_config
from arxiv_digest.delivery.feishu import FeishuDelivery


def build_parser():
    parser = argparse.ArgumentParser(
        description="Resend an existing digest without fetching papers or calling an LLM"
    )
    parser.add_argument("digest", type=Path)
    parser.add_argument("--feishu", action="store_true", help="Send through Feishu webhook")
    parser.add_argument("--force", action="store_true", help="Ignore matching delivery receipt")
    parser.add_argument("--dry-run", action="store_true", help="Build messages without sending")
    return parser


def main(argv=None, delivery_factory=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.feishu:
        parser.error("select a delivery target with --feishu")
    if not args.digest.is_file():
        parser.error(f"digest not found: {args.digest}")

    digest = args.digest.read_text(encoding="utf-8")
    _, preferences = load_project_config(ROOT)
    factory = delivery_factory or FeishuDelivery.from_env
    sender = factory(preferences.get("delivery", {}).get("feishu", {}))
    try:
        result = sender.send_markdown(
            date=args.digest.stem,
            digest=digest,
            force=args.force,
            dry_run=args.dry_run,
        )
    finally:
        sender.close()
    if not result.success:
        parser.error(result.error or "delivery failed")
    status = "skipped duplicate" if result.skipped else f"{result.message_count} messages"
    print(f"[FEISHU] {status}")
    return result


if __name__ == "__main__":
    main()
