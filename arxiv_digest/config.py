from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_project_config(root: Path = PROJECT_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    topics = load_json(root / "config" / "topics.json")
    preferences = load_json(root / "config" / "preferences.json")
    return topics, preferences
