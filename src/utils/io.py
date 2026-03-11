from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False)


def write_text(path: str | Path, payload: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(payload)


def _table_fallback_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_suffix(target.suffix + ".pkl")


def _table_meta_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_suffix(target.suffix + ".meta.json")


def read_parquet(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    try:
        if target.exists():
            return pd.read_parquet(target)
    except Exception:
        pass
    fallback = _table_fallback_path(target)
    if fallback.exists():
        return pd.read_pickle(fallback)
    raise FileNotFoundError(f"No parquet or explicit fallback found for {target}")


def write_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fallback = _table_fallback_path(target)
    meta_path = _table_meta_path(target)
    try:
        frame.to_parquet(target, index=False)
        if fallback.exists():
            fallback.unlink()
        if meta_path.exists():
            meta_path.unlink()
    except Exception:
        if target.exists():
            target.unlink()
        frame.to_pickle(fallback)
        write_json(
            meta_path,
            {
                "requested_path": str(target),
                "actual_format": "pickle",
                "actual_path": str(fallback),
            },
        )


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
