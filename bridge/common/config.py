"""Load bridge configuration with local deployment settings."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
import yaml

BRIDGE_ROOT = Path(__file__).resolve().parents[1]
_ENV_REF = re.compile(r"\$\{(WBWAM_BRIDGE_[A-Z0-9_]+)\}")


def _expand_bridge_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_bridge_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_bridge_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        result = os.environ.get(name)
        if not result:
            raise ValueError(f"Set {name} in bridge/.env or the environment")
        return result

    return _ENV_REF.sub(replace, value)


def read_deployment_config(path: Path) -> dict[str, Any]:
    """Read YAML after applying bridge/.env; exported variables take precedence."""
    load_dotenv(BRIDGE_ROOT / ".env", override=False)
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return _expand_bridge_env(payload)
