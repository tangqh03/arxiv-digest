import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def digest_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_digest.py"
    spec = importlib.util.spec_from_file_location("generate_digest", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
