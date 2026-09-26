"""Source discovery, fingerprints and non-destructive output helpers."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

from .schema import FPS, VERSION


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}


def contained_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError(f"Expected a relative path under {root}: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes its source/output directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_frames(episode: Path) -> list[dict]:
    payload = read_json(episode / "data.json")
    frames = payload.get("frames") if isinstance(payload, dict) else payload
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError(f"{episode}: expected at least two frames")
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("frame_index") != index:
            raise ValueError(f"{episode}: frame_index must be contiguous from zero; frame {index}")
    return frames


def image_paths(episode: Path, frames: list[dict], camera: str) -> list[Path]:
    paths = []
    for frame in frames[:-1]:
        if frame.get("primary_camera") != camera:
            raise ValueError(f"{episode}: inconsistent primary_camera; expected {camera}")
        color = frame.get("cameras", {}).get(camera, {}).get("color")
        paths.append(contained_path(episode, color))
    return paths


def canonical_task(name: str) -> str:
    text = re.sub(r"_\d{6}(?:_\d+)?$", "", str(name).strip())
    text = " ".join(text.replace("_", " ").split())
    if not text:
        raise ValueError("Task text must not be empty")
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def discover(source: Path, *, task=None, task_id=None, exclude=(), limit=None) -> list[dict]:
    if not source.is_dir():
        raise NotADirectoryError(source)
    task_dirs = (
        [source]
        if (source / "metadata.json").is_file()
        else sorted(p for p in source.iterdir() if p.is_dir() and (p / "metadata.json").is_file())
    )
    if not task_dirs:
        raise ValueError("No collector tasks found; expected metadata.json and episode_*/data.json")
    if (task is not None or task_id is not None) and len(task_dirs) != 1:
        raise ValueError("--task and --task-id require a single input task")
    exclusions, matched = set(exclude), set()
    jobs, counters, task_ids = [], {}, {}
    for directory in task_dirs:
        metadata = read_json(directory / "metadata.json")
        if metadata.get("capture_fps") != FPS:
            raise ValueError(f"{directory}: capture_fps must be explicitly {FPS}; no automatic resampling")
        camera = metadata.get("primary_camera")
        if not isinstance(camera, str) or not camera:
            raise ValueError(f"{directory}: missing primary_camera")
        prompt = canonical_task(task or metadata.get("task_name") or directory.name)
        key = task_id or re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
        if not key or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", key):
            raise ValueError("Use --task-id with a lowercase ASCII task directory name")
        if key in task_ids and task_ids[key] != prompt:
            raise ValueError(f"Different task texts map to the same directory {key!r}")
        task_ids[key] = prompt
        episodes = []
        for path in sorted(directory.glob("episode_*")):
            if not path.is_dir() or not re.fullmatch(r"episode_\d+", path.name):
                raise ValueError(f"Invalid episode directory: {path}")
            aliases = {f"{directory.name}/{path.name}"}
            if len(task_dirs) == 1:
                aliases.add(path.name)
            if aliases & exclusions:
                matched.update(aliases & exclusions)
            else:
                episodes.append(path)
        episodes.sort(key=lambda p: int(p.name.split("_")[-1]))
        episodes = episodes[:limit] if limit is not None else episodes
        if not episodes:
            raise ValueError(f"No episodes selected under {directory}")
        counters[key] = counters.get(key, 0) + 1
        jobs.append(
            {
                "source": directory,
                "episodes": episodes,
                "camera": camera,
                "task": prompt,
                "relative_output": f"{key}/record_{counters[key]:04d}",
                "capture_mode": metadata.get("capture_mode", "unspecified"),
            }
        )
    if exclusions - matched:
        raise ValueError(f"Unknown --exclude-episode selections: {sorted(exclusions - matched)}")
    return jobs


def source_signature(job: dict, options: dict) -> str:
    hashes = {"metadata.json": file_hash(job["source"] / "metadata.json")}
    for episode in job["episodes"]:
        hashes[f"{episode.name}/data.json"] = file_hash(episode / "data.json")
        frames = load_frames(episode)
        for path in set(image_paths(episode, frames, job["camera"])):
            hashes[f"{episode.name}/{path.relative_to(episode.resolve())}"] = file_hash(path)
    return digest(
        {
            "version": VERSION,
            "task": job["task"],
            "source": job["source"].name,
            "relative_output": job["relative_output"],
            "options": options,
            "files": hashes,
        }
    )


def check_output_location(source: Path, output: Path) -> None:
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Input and output must be separate, non-overlapping directories")
    if output in (Path.home().resolve(), Path.cwd().resolve(), Path(output.anchor)):
        raise ValueError("Refusing a home, working-directory or filesystem root as output")


@contextmanager
def output_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".conversion.lock"
    # O_NOFOLLOW prevents a pre-existing lock symlink from touching another file.
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another converter is using this output directory") from exc
        yield
