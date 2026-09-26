from pathlib import Path

PATH_STORAGE = Path("storage")

PATH_ASSET = PATH_STORAGE / "assets"

PATH_CKPT = PATH_STORAGE / "ckpts"

PATH_LOG = PATH_STORAGE / "logs"


def get_path_log(tag):
    return PATH_LOG / tag


def get_latest_ckpt(tag):
    ckpt_dir = PATH_LOG / tag / "checkpoints"
    ckpts = [
        ckpt for ckpt in Path(ckpt_dir).glob("*") if not ckpt.name.endswith(".json")
    ]
    ckpts.sort(key=lambda x: int(x.name))
    return ckpts[-1] if ckpts else None
