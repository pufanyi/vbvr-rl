from __future__ import annotations

import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pixi_is_the_only_project_environment_workflow():
    manifest = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pixi = manifest["tool"]["pixi"]

    assert pixi["workspace"]["requires-pixi"] == ">=0.77.0,<0.78"
    assert pixi["workspace"]["platforms"] == [
        {
            "name": "linux-64-cuda-12-6",
            "platform": "linux-64",
            "cuda": "12.6",
        },
        "linux-64",
    ]
    assert pixi["feature"]["cuda"]["platforms"] == ["linux-64-cuda-12-6"]
    assert pixi["feature"]["lint"]["platforms"] == ["linux-64"]
    assert pixi["feature"]["vllm"]["platforms"] == ["linux-64-cuda-12-6"]
    assert {"default", "lint", "vllm"} <= set(pixi["environments"])
    assert pixi["environments"]["default"]["features"] == ["dev", "cuda"]
    assert pixi["environments"]["vllm"]["features"] == ["vllm", "cuda"]
    assert pixi["tasks"]["test"] == "python -m pytest tests"
    assert (_REPO_ROOT / "pixi.lock").is_file()
    assert not (_REPO_ROOT / "uv.lock").exists()
    assert not (_REPO_ROOT / ".python-version").exists()


def test_release_sources_do_not_reference_legacy_environment_commands():
    forbidden = (
        ".venv/bin/",
        "[tool.uv",
        "uv pip",
        "uv run",
        "uv sync",
        "astral-sh/ruff-action",
    )
    roots = (
        _REPO_ROOT / ".github",
        _REPO_ROOT / "configs",
        _REPO_ROOT / "docs",
        _REPO_ROOT / "scripts",
        _REPO_ROOT / "src",
    )
    paths = [
        _REPO_ROOT / "AGENTS.md",
        _REPO_ROOT / "CONTRIBUTING.md",
        _REPO_ROOT / "README.md",
        _REPO_ROOT / "pyproject.toml",
    ]
    paths.extend(path for root in roots for path in root.rglob("*") if path.is_file())

    violations: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(_REPO_ROOT)}: {token}")

    assert not violations, f"legacy environment workflow references: {violations}"
