from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def merge_yaml_files(paths: list[str | Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, load_yaml(path))
    return merged


def resolve_relative_paths(config: Any, root: str | Path) -> Any:
    base = Path(root)
    if isinstance(config, dict):
        return {key: resolve_relative_paths(value, base) for key, value in config.items()}
    if isinstance(config, list):
        return [resolve_relative_paths(value, base) for value in config]
    if isinstance(config, str) and (config.startswith("./") or config.startswith("../")):
        return str((base / config).resolve())
    return config


def load_config_bundle(
    defaults_path: str | Path,
    paths_path: str | Path,
    extra_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    pieces = [defaults_path, paths_path]
    if extra_paths:
        pieces.extend(extra_paths)
    merged = merge_yaml_files(pieces)
    return resolve_relative_paths(merged, Path(defaults_path).parent)

