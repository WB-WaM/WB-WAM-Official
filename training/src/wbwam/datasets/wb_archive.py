"""Discover per-record WB datasets inside sealed release archives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


def _normalize_include_tasks(value: Any, *, archive_name: str) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise TypeError(f"WB archive {archive_name!r} `include_tasks` must be a non-empty list of names.")

    tasks = [str(task).strip() for task in value]
    if not tasks or any(not task for task in tasks):
        raise ValueError(f"WB archive {archive_name!r} `include_tasks` must contain non-empty names.")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"WB archive {archive_name!r} `include_tasks` contains duplicate names.")
    return frozenset(tasks)


def expand_wb_archive_configs(archives: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand archive-level configs into deterministic per-record configs."""

    datasets: list[dict[str, Any]] = []
    archive_names: set[str] = set()
    archive_roots: set[Path] = set()
    dataset_names: set[str] = set()

    for index, raw_archive in enumerate(archives):
        if not isinstance(raw_archive, Mapping):
            raise TypeError(f"WB archive config at index {index} must be a mapping.")

        archive = deepcopy(dict(raw_archive))
        try:
            name = str(archive.pop("name")).strip()
            root = Path(str(archive.pop("root"))).expanduser()
        except KeyError as exc:
            raise ValueError(f"WB archive config at index {index} is missing `{exc.args[0]}`.") from exc

        if not name:
            raise ValueError(f"WB archive config at index {index} has an empty name.")
        include_tasks = _normalize_include_tasks(archive.pop("include_tasks", None), archive_name=name)
        if name in archive_names:
            raise ValueError(f"Duplicate WB archive name: {name}")
        comparable_root = root.resolve(strict=False)
        if comparable_root in archive_roots:
            raise ValueError(f"Duplicate WB archive root: {root}")
        if not root.is_dir():
            raise FileNotFoundError(f"WB archive root does not exist: {root}")

        info_paths = sorted(root.glob("*/record_*/meta/info.json"), key=lambda path: path.as_posix())
        if not info_paths:
            raise ValueError(f"WB archive contains no <task>/record_*/meta/info.json entries: {root}")
        available_tasks = {info_path.relative_to(root).parts[0] for info_path in info_paths}
        if include_tasks is not None:
            unknown_tasks = sorted(include_tasks - available_tasks)
            if unknown_tasks:
                raise ValueError(
                    f"WB archive {name!r} requested unknown tasks {unknown_tasks}; "
                    f"available tasks: {sorted(available_tasks)}"
                )
            info_paths = [
                info_path for info_path in info_paths if info_path.relative_to(root).parts[0] in include_tasks
            ]

        archive_names.add(name)
        archive_roots.add(comparable_root)
        for info_path in info_paths:
            record_root = info_path.parent.parent
            relative_root = record_root.relative_to(root)
            if len(relative_root.parts) != 2:
                raise ValueError(f"Unexpected WB record layout under {root}: {record_root}")

            dataset_name = f"{name}/{relative_root.as_posix()}"
            if dataset_name in dataset_names:
                raise ValueError(f"Duplicate WB dataset name: {dataset_name}")

            dataset = deepcopy(archive)
            dataset.update(name=dataset_name, root=str(record_root))
            datasets.append(dataset)
            dataset_names.add(dataset_name)

    if not datasets:
        raise ValueError("WB archive list must be non-empty.")
    return datasets


def resolve_wb_dataset_configs(
    *,
    archives: Iterable[Mapping[str, Any]] | None = None,
    datasets: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Resolve either archive configs or explicit dataset configs."""

    if archives is not None and datasets is not None:
        raise ValueError("WB data config accepts either `archives` or `datasets`, not both.")
    if archives is not None:
        return expand_wb_archive_configs(archives)
    if datasets is None:
        raise ValueError("WB data config requires a non-empty `archives` or `datasets` list.")

    resolved = []
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, Mapping):
            raise TypeError(f"WB dataset config at index {index} must be a mapping.")
        resolved.append(deepcopy(dict(dataset)))
    if not resolved:
        raise ValueError("WB dataset list must be non-empty.")
    return resolved
