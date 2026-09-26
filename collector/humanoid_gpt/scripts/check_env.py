from __future__ import annotations

import importlib
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED = ("numpy", "zmq", "cv2")
OPTIONAL = ("pyrealsense2",)


def main() -> int:
    failures = 0
    for module in REQUIRED:
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "available")
            print(f"[OK]   import {module}: {version}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] import {module}: {exc}")
    for module in OPTIONAL:
        try:
            importlib.import_module(module)
            print(f"[OK]   import {module}")
        except Exception as exc:
            print(f"[WARN] import {module}: {exc} (only required for local cameras)")
    try:
        from collector.humanoid_gpt.core.codec import decode_topic_message
        from collector.humanoid_gpt.core.episode_writer import EpisodeWriter
        from collector.humanoid_gpt.core.subscriber import SubscriberHub

        assert decode_topic_message and EpisodeWriter and SubscriberHub
        print("[OK]   HGPT collector modules")
    except Exception as exc:
        failures += 1
        print(f"[FAIL] HGPT collector modules: {exc}")
    print(f"HGPT collector environment check: failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
