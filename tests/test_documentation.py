from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LINK_RE = re.compile(r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+[^)]*)?\)")


def _release_documents() -> list[Path]:
    return [
        *sorted(_REPO_ROOT.glob("*.md")),
        *sorted((_REPO_ROOT / "docs").rglob("*.md")),
        *sorted((_REPO_ROOT / "scripts").rglob("*.md")),
    ]


@pytest.mark.parametrize("document", _release_documents(), ids=lambda path: str(path.relative_to(_REPO_ROOT)))
def test_release_document_local_links_resolve(document: Path):
    missing: list[str] = []
    for match in _LINK_RE.finditer(document.read_text(encoding="utf-8")):
        raw_target = match.group("target").removeprefix("<").removesuffix(">")
        parsed = urlsplit(raw_target)
        if parsed.scheme or raw_target.startswith("#"):
            continue
        relative = unquote(parsed.path)
        if relative and not (document.parent / relative).resolve().exists():
            missing.append(raw_target)
    assert not missing, f"broken local links in {document.relative_to(_REPO_ROOT)}: {missing}"


def test_release_documents_do_not_expose_environment_specific_details():
    forbidden = (
        "fujian",
        "福建",
        "malaysia",
        "马来",
        "aoss",
        "/mnt/umm/",
        "/mnt/aigc/",
    )
    violations: list[str] = []
    for document in _release_documents():
        text = document.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token.lower() in text:
                violations.append(f"{document.relative_to(_REPO_ROOT)}: {token}")
    assert not violations, f"environment-specific release documentation: {violations}"


def test_only_vbvr_pro_evaluation_surface_is_shipped():
    assert not (_REPO_ROOT / "third_party/VBVR-EvalKit").exists()
    assert not (_REPO_ROOT / "scripts/eval/vbvr").exists()
    assert not (_REPO_ROOT / "src/eval/vbvr").exists()
    assert not (_REPO_ROOT / "src/cli/eval_vbvr.py").exists()
