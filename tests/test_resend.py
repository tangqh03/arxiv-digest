import importlib.util
from pathlib import Path

from arxiv_digest.delivery.feishu import DeliveryResult


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "resend_daily_digest.py"


def load_script():
    spec = importlib.util.spec_from_file_location("resend_daily_digest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDelivery:
    def __init__(self):
        self.calls = []
        self.closed = False

    def send_markdown(self, **kwargs):
        self.calls.append(kwargs)
        return DeliveryResult(True, 1)

    def close(self):
        self.closed = True


def test_resend_uses_existing_digest_without_pipeline(tmp_path):
    digest_path = tmp_path / "2026-09-03.md"
    digest_path.write_text("# Existing Digest\n\nSaved content", encoding="utf-8")
    sender = FakeDelivery()

    result = load_script().main(
        [str(digest_path), "--feishu", "--force"],
        delivery_factory=lambda config: sender,
    )

    assert result.success is True
    assert sender.calls == [
        {
            "date": "2026-09-03",
            "digest": "# Existing Digest\n\nSaved content",
            "force": True,
            "dry_run": False,
        }
    ]
    assert sender.closed is True
