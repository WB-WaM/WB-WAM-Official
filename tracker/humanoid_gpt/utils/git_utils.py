"""Lightweight git introspection helpers for run tracking / experiment logging.

All helpers fail silently and return ``"unknown"`` (or ``None`` / ``False`` for
booleans) when git is unavailable, the working tree is not a git repo, or the
subprocess call errors out for any reason.  This guarantees that experiment
logging never crashes on machines without git installed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_git(args: list[str], cwd: Path = _REPO_ROOT, timeout: float = 2.0) -> Optional[str]:
    """Run a git command and return stripped stdout, or ``None`` on any failure."""
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return out.decode("utf-8", errors="replace").strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def get_git_commit(short: bool = False) -> str:
    """Return current commit SHA. Short (7 chars) or full. Falls back to ``"unknown"``."""
    rev = _run_git(["rev-parse", "--short" if short else "HEAD"] if short else ["rev-parse", "HEAD"])
    return rev or "unknown"


def get_git_branch() -> str:
    """Return current branch name; ``"detached"`` if HEAD is detached."""
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch is None:
        return "unknown"
    return branch if branch != "HEAD" else "detached"


def is_git_dirty() -> bool:
    """Return True if the working tree has uncommitted changes (tracked files)."""
    status = _run_git(["status", "--porcelain", "--untracked-files=no"])
    return bool(status)


def get_git_remote(remote: str = "origin") -> Optional[str]:
    """Return remote URL, or ``None`` if unavailable."""
    return _run_git(["config", "--get", f"remote.{remote}.url"])


def get_git_info() -> dict:
    """Collect a flat dict of git metadata suitable for swanlab/wandb config.

    All keys are always present; missing values fall back to sentinels so
    downstream loggers don't have to special-case anything.
    """
    full = get_git_commit(short=False)
    return {
        "git_commit": full,
        "git_commit_short": full[:7] if full != "unknown" else "unknown",
        "git_branch": get_git_branch(),
        "git_dirty": is_git_dirty(),
        "git_remote": get_git_remote() or "unknown",
    }
