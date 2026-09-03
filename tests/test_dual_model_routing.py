from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_daily_cli_routes_screening_and_deep_models_separately():
    source = (ROOT / "scripts" / "run_daily.py").read_text(encoding="utf-8")

    assert 'env_prefix="SCREENING_LLM"' in source
    assert 'env_prefix="DEEP_LLM"' in source
    assert "PaperScreener(\n            screening_llm" in source
    assert "DeepSummarizer(\n            deep_llm" in source


def test_agent_is_the_configured_topic():
    import json

    topics = json.loads((ROOT / "config" / "topics.json").read_text(encoding="utf-8"))

    assert "agent" in topics["topics"]
    assert all(isinstance(topic, str) for topic in topics["topics"])
