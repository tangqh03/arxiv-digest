import base64
import hashlib
import hmac
import json

import httpx

from arxiv_digest.delivery.feishu import FeishuDelivery, generate_signature
from arxiv_digest.models import Paper
from arxiv_digest.rendering.markdown import DigestStats


def stats(count=1):
    return DigestStats(8, 6, count, min(count, 1))


def paper(index=1, *, long=False):
    text = ("很长的摘要" * 2000) if long else "中文摘要"
    return Paper(
        arxiv_id=f"2609.{index:05d}",
        title=f"Paper {index}",
        abstract_zh=text,
        tldr_zh=f"TLDR {index}",
        screening={
            "relevance": 10 - index,
            "novelty": 8,
            "impact": 7,
            "relevance_reason": "直接相关",
        },
        abs_url=f"https://arxiv.org/abs/2609.{index:05d}",
        hf_url=f"https://huggingface.co/papers/2609.{index:05d}",
    )


def delivery(tmp_path, handler=None, **kwargs):
    handler = handler or (lambda request: httpx.Response(200, json={"code": 0}))
    return FeishuDelivery(
        webhook_url="https://open.feishu.test/hook/redacted",
        receipt_root=tmp_path,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def payload_text(payload):
    return payload["card"]["elements"][0]["text"]["content"]


def test_feishu_builds_valid_payload(tmp_path):
    payload = delivery(tmp_path).build_payloads(
        date="2026-09-03", papers=[paper()], stats=stats()
    )[0]

    assert payload["msg_type"] == "interactive"
    assert payload["card"]["header"]["title"]["tag"] == "plain_text"
    assert payload["card"]["elements"][0]["text"]["tag"] == "lark_md"


def test_feishu_card_contains_title(tmp_path):
    payloads = delivery(tmp_path).build_payloads(
        date="2026-09-03", papers=[paper()], stats=stats()
    )

    assert "Paper 1" in payload_text(payloads[1])


def test_feishu_custom_card_title(tmp_path):
    payload = delivery(tmp_path).build_payloads(
        date="2026-09-03",
        papers=[],
        stats=stats(0),
        card_title="[TEST] arxiv-digest Feishu integration",
        overview_note="这是一条自动化测试消息，可以忽略。",
    )[0]

    assert payload["card"]["header"]["title"]["content"] == (
        "[TEST] arxiv-digest Feishu integration"
    )
    assert "这是一条自动化测试消息，可以忽略。" in payload_text(payload)


def test_feishu_card_contains_links(tmp_path):
    payloads = delivery(tmp_path).build_payloads(
        date="2026-09-03", papers=[paper()], stats=stats()
    )

    assert "https://arxiv.org/abs/2609.00001" in payload_text(payloads[1])
    assert "https://huggingface.co/papers/2609.00001" in payload_text(payloads[1])


def test_feishu_chunks_by_paper(tmp_path):
    payloads = delivery(tmp_path, max_papers_per_message=2).build_payloads(
        date="2026-09-03", papers=[paper(i) for i in range(1, 6)], stats=stats(5)
    )

    assert len(payloads) == 4  # overview plus three paper groups
    assert "Paper 1" in payload_text(payloads[1])
    assert "Paper 3" in payload_text(payloads[2])
    assert "Paper 5" in payload_text(payloads[3])


def test_feishu_does_not_split_single_paper_midway(tmp_path):
    payloads = delivery(tmp_path, max_papers_per_message=1).build_payloads(
        date="2026-09-03", papers=[paper(long=True)], stats=stats()
    )

    assert len(payloads) == 2
    assert payload_text(payloads[1]).count("很长的摘要") == 2000


def test_feishu_success(tmp_path):
    result = delivery(tmp_path).send(
        date="2026-09-03", digest="digest", papers=[paper()], stats=stats()
    )

    assert result.success is True
    assert result.message_count == 2


def test_feishu_api_error(tmp_path):
    sender = delivery(
        tmp_path, handler=lambda request: httpx.Response(200, json={"code": 19001})
    )

    result = sender.send(
        date="2026-09-03", digest="digest", papers=[paper()], stats=stats()
    )

    assert result.success is False
    assert result.error == "Feishu API error code 19001"


def test_feishu_http_error(tmp_path):
    sender = delivery(
        tmp_path, handler=lambda request: httpx.Response(500, json={"code": 1})
    )

    result = sender.send(
        date="2026-09-03", digest="digest", papers=[paper()], stats=stats()
    )

    assert result.success is False
    assert result.error == "Feishu HTTP 500"


def test_feishu_signature_generation():
    timestamp = "1599360473"
    secret = "test-secret"
    expected = base64.b64encode(
        hmac.new(f"{timestamp}\n{secret}".encode(), digestmod=hashlib.sha256).digest()
    ).decode()

    assert generate_signature(timestamp, secret) == expected


def test_feishu_secret_not_logged(tmp_path, capsys, caplog):
    secret = "do-not-print-this-secret"
    sender = delivery(
        tmp_path,
        secret=secret,
        handler=lambda request: httpx.Response(500, json={"code": 1}),
    )

    result = sender.send(
        date="2026-09-03", digest="digest", papers=[paper()], stats=stats()
    )

    captured = capsys.readouterr()
    assert secret not in f"{captured.out}{captured.err}{caplog.text}{result.error}"


def test_delivery_receipt_written(tmp_path):
    delivery(tmp_path).send(
        date="2026-09-03", digest="digest", papers=[paper()], stats=stats()
    )

    receipt = json.loads(
        (tmp_path / "2026-09-03" / "feishu.json").read_text(encoding="utf-8")
    )
    assert receipt["success"] is True
    assert receipt["message_count"] == 2
    assert "secret" not in receipt
    assert "webhook" not in receipt


def test_same_digest_not_sent_twice(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"code": 0})

    sender = delivery(tmp_path, handler=handler)
    sender.send(date="2026-09-03", digest="same", papers=[paper()], stats=stats())
    second = sender.send(
        date="2026-09-03", digest="same", papers=[paper()], stats=stats()
    )

    assert calls == 2
    assert second.skipped is True
    assert second.message_count == 0


def test_force_delivery_resends(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"code": 0})

    sender = delivery(tmp_path, handler=handler)
    sender.send(date="2026-09-03", digest="same", papers=[paper()], stats=stats())
    second = sender.send(
        date="2026-09-03",
        digest="same",
        papers=[paper()],
        stats=stats(),
        force=True,
    )

    assert calls == 4
    assert second.success is True
    assert second.skipped is False


def test_markdown_resend_chunks_at_paper_boundaries(tmp_path):
    digest = "# Digest\n\nOverview\n\n### Paper One\n\nBody one\n\n### Paper Two\n\nBody two"
    sender = delivery(tmp_path, max_papers_per_message=1)

    payloads = sender.build_markdown_payloads(date="2026-09-03", digest=digest)

    assert len(payloads) == 3
    assert "Paper One" in payload_text(payloads[1])
    assert "Paper Two" not in payload_text(payloads[1])
    assert "Paper Two" in payload_text(payloads[2])
