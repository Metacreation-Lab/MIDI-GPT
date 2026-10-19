"""Every place that states the package version agrees with midigpt.__version__."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import midigpt

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("path, pattern", [
    ("pyproject.toml", r'(?m)^version\s*=\s*"([^"]+)"'),
    ("CITATION.cff", r"(?m)^version:\s*(\S+)"),
    ("CMakeLists.txt", r"project\(midigpt VERSION (\S+)"),
])
def test_version_matches(path, pattern):
    match = re.search(pattern, (ROOT / path).read_text())
    assert match, f"no version in {path}"
    assert match.group(1) == midigpt.__version__, f"{path} says {match.group(1)}"


def test_http_api_reports_the_package_version(tiny_gpt2, ghost_tokenizer, ghost_analyzer):
    from midigpt.http.server import HttpServer, ModelEntry
    from midigpt.inference.engine import InferenceEngine

    engine = InferenceEngine(tiny_gpt2, ghost_tokenizer, ghost_analyzer)
    server = HttpServer({"m": ModelEntry(id="m", engine=engine, label="m")}, default_model="m")
    assert server.app.version == midigpt.__version__
