"""HGPT-local loader for the existing, unchanged MANUS provider."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
MANUS_BINDING_DIR = PROJECT_ROOT / ".deps" / "manus_sdk"
SONIC_ROOT = REPO_ROOT / "tracker" / "sonic"

for path in (MANUS_BINDING_DIR, SONIC_ROOT):
    value = str(path)
    if path.is_dir() and value not in sys.path:
        sys.path.insert(0, value)

try:
    from gear_sonic.scripts.manus_provider import (  # noqa: F401
        DEFAULT_MANUS_CALIBRATION_FILE,
        DEFAULT_MANUS_INIT_TIMEOUT_S,
        DEFAULT_MANUS_LEFT_CALIBRATION_FILE,
        DEFAULT_MANUS_MCP_JOINT,
        DEFAULT_MANUS_RIGHT_CALIBRATION_FILE,
        DEFAULT_MANUS_STALE_S,
        VALID_MANUS_MCP_JOINTS,
        ManusIntegratedProvider,
    )
except Exception as exc:  # pragma: no cover - native SDK dependent
    raise RuntimeError(
        "MANUS provider is unavailable. Run scripts/setup_pico_env.sh --with-manus "
        "to build the Python binding in tracker/humanoid_gpt/.deps/manus_sdk."
    ) from exc

__all__ = [
    "DEFAULT_MANUS_CALIBRATION_FILE",
    "DEFAULT_MANUS_INIT_TIMEOUT_S",
    "DEFAULT_MANUS_LEFT_CALIBRATION_FILE",
    "DEFAULT_MANUS_MCP_JOINT",
    "DEFAULT_MANUS_RIGHT_CALIBRATION_FILE",
    "DEFAULT_MANUS_STALE_S",
    "ManusIntegratedProvider",
    "VALID_MANUS_MCP_JOINTS",
]
