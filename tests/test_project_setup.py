from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_env_example_has_no_real_secrets():
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, value = line.split("=", 1)
            values[name] = value

    assert values
    assert all(value == "" for value in values.values())
    assert {
        "LLM_API_KEY",
        "SCREENING_LLM_MODEL",
        "DEEP_LLM_MODEL",
        "FEISHU_WEBHOOK_URL",
        "FEISHU_TEST_WEBHOOK_URL",
    } <= values.keys()


def test_daily_shell_uses_uv_and_strict_mode():
    script = (ROOT / "scripts" / "run_daily.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "uv run python scripts/run_daily.py" in script


def test_ci_excludes_live_tests():
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )

    assert 'python-version: ["3.11", "3.12"]' in workflow
    assert 'pytest -m "not live"' in workflow


def test_daily_workflow_runs_at_utc_plus_8_ten_am():
    workflow = (ROOT / ".github" / "workflows" / "daily-digest.yml").read_text(
        encoding="utf-8"
    )

    assert 'cron: "0 2 * * *"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "TZ: Asia/Singapore" in workflow
    assert "--max-papers 50" in workflow
    assert "--deep-top-k 3" in workflow
    assert "--deliver feishu" in workflow
    assert "default: false" in workflow
    assert "args+=(--dry-run)" in workflow


def test_daily_workflow_uses_secrets_and_persistent_receipts():
    workflow = (ROOT / ".github" / "workflows" / "daily-digest.yml").read_text(
        encoding="utf-8"
    )

    assert "secrets.SCREENING_LLM_API_KEY" in workflow
    assert "secrets.DEEP_LLM_API_KEY" in workflow
    assert "secrets.FEISHU_WEBHOOK_URL" in workflow
    assert "memory/analysis" in workflow
    assert "memory/deep_summaries" in workflow
    assert "memory/fulltext" in workflow
    assert "memory/delivery" in workflow


def test_project_feishu_config_uses_one_card_per_paper():
    import json

    preferences = json.loads(
        (ROOT / "config" / "preferences.json").read_text(encoding="utf-8")
    )

    assert preferences["delivery"]["feishu"]["max_papers_per_message"] == 1
