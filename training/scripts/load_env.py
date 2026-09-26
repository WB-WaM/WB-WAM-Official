"""Load training/.env and emit safely quoted exports for Bash/Zsh."""

import os
from pathlib import Path
import re
import shlex

from dotenv import dotenv_values, load_dotenv

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = {
    "WB_WAM_DATA_ROOT", "WB_WAM_PICO_ROOT", "WB_WAM_REAL_ROOT",
    "DIFFSYNTH_MODEL_BASE_PATH", "WBWAM_ACTION_DIT_PATH", "WBWAM_TEXT_CACHE_ROOT",
    "WBWAM_RUNS_ROOT", "WBWAM_STATS_ROOT", "WBWAM_PRETRAIN_CHECKPOINT",
    "WBWAM_ARENA_NATIVE50_ROOT",
    "WBWAM_MIDTRAIN_CHECKPOINT", "WBWAM_CACHE_ROOT", "HF_HOME", "TORCH_HOME",
    "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "WANDB_DIR", "PIP_CACHE_DIR",
    "TMPDIR", "CUDA_HOME", "WBWAM_CUDA_HOME_DEFAULT", "WBWAM_FFMPEG_ROOT_DEFAULT",
    "FFMPEG_ROOT", "WBWAM_FONT_DIR",
}


def main():
    env_file = TRAINING_ROOT / ".env"
    if not env_file.is_file():
        raise FileNotFoundError("Copy training/.env.example to training/.env and configure your paths.")

    keys = set(dotenv_values(env_file, interpolate=False))
    for key in keys:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid environment variable name: {key!r}")
    # Preserve the shared-root override supported by the training launchers.
    if os.environ.get("WB_WAM_DATA_ROOT"):
        for key, name in (("WB_WAM_PICO_ROOT", "pico_archive"), ("WB_WAM_REAL_ROOT", "real_archive")):
            os.environ.setdefault(key, str(Path(os.environ["WB_WAM_DATA_ROOT"]) / name))
    load_dotenv(env_file, override=False)
    for key in keys & PATH_KEYS:
        if os.environ.get(key):
            path = Path(os.environ[key]).expanduser()
            os.environ[key] = str(path if path.is_absolute() else TRAINING_ROOT / path)
    os.environ["WBWAM_TRAINING_ROOT"] = str(TRAINING_ROOT)
    source = str(TRAINING_ROOT / "src")
    python_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if source not in python_path:
        os.environ["PYTHONPATH"] = os.pathsep.join([source, *filter(None, python_path)])
    keys.update(("WBWAM_TRAINING_ROOT", "PYTHONPATH"))
    for key in sorted(keys):
        if key in os.environ:
            print(f"export {key}={shlex.quote(os.environ[key])}")


if __name__ == "__main__":
    main()
