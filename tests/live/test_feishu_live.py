import os

import pytest

from arxiv_digest.delivery.feishu import FeishuDelivery
from arxiv_digest.rendering.markdown import DigestStats


pytestmark = pytest.mark.live


def test_feishu_test_webhook_live(tmp_path):
    webhook = os.environ.get("FEISHU_TEST_WEBHOOK_URL")
    if not webhook:
        pytest.skip("FEISHU_TEST_WEBHOOK_URL is not configured")
    sender = FeishuDelivery(
        webhook_url=webhook,
        secret=os.environ.get("FEISHU_TEST_WEBHOOK_SECRET") or None,
        receipt_root=tmp_path,
    )
    try:
        result = sender.send(
            date="integration-test",
            digest="feishu-live-integration-test",
            papers=[],
            stats=DigestStats(0, 0, 0, 0),
            force=True,
            card_title="[TEST] arxiv-digest Feishu integration",
            overview_note="这是一条自动化测试消息，可以忽略。",
        )
    finally:
        sender.close()

    assert result.success is True
    assert result.message_count == 1
