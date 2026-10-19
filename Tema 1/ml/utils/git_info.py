"""Git commit + dirty flag (recorded in every run)."""
from __future__ import annotations

import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_provenance(cwd: Path | None = None) -> dict:
    root = cwd or repo_root()
    # Prefer stamped files when the checkout has no .git (e.g. preprocess VM)
    for base in (root / "ml", root):
        cpath, dpath = base / "COMMIT", base / "CODE_DIRTY"
        if cpath.exists():
            commit = cpath.read_text().strip()
            dirty = dpath.read_text().strip().lower() == "true" if dpath.exists() else False
            return {"commit": commit, "dirty": dirty}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "", "dirty": True}


def format_git_tag(prov: dict | None = None) -> str:
    p = prov or git_provenance()
    return f"{p['commit']}{'-dirty' if p['dirty'] else ''}"
