from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src.utils.audio import probe_audio_many
from src.utils.io import write_csv, write_json, write_parquet


REQUIRED_SPLITS = ("train", "val", "test")


def _snake_case(name: str) -> str:
    normalized = re.sub(r"\s+", " ", (name or "").strip().lower())
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"[^a-z0-9 ]+", "", normalized)
    return normalized.replace(" ", "_")


def canonicalize_qs_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {column: _snake_case(column) for column in frame.columns}
    return frame.rename(columns=rename_map)


def parse_qualispeech_split(
    root_dir: str | Path,
    split_name: str,
    *,
    audio_cache_path: str | Path | None = None,
    audio_probe_workers: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root_path = Path(root_dir)
    csv_path = root_path / f"{split_name}.csv"
    audio_dir = root_path / split_name
    frame = canonicalize_qs_columns(pd.read_csv(csv_path))
    if "id" not in frame.columns:
        raise ValueError(f"QualiSpeech CSV missing 'id' column: {csv_path}")
    rows: list[dict[str, Any]] = []
    missing_audio_paths: list[str] = []
    audio_paths_to_probe: list[str] = []
    for record in tqdm(
        frame.to_dict(orient="records"),
        desc=f"QualiSpeech {split_name}",
        unit="utt",
    ):
        filename = record["id"]
        audio_path = audio_dir / filename
        audio_path_str = str(audio_path)
        if not audio_path.exists():
            missing_audio_paths.append(audio_path_str)
        else:
            audio_paths_to_probe.append(audio_path_str)
        row = {
            "dataset": "qualispeech",
            "split_original": split_name,
            "utt_id": Path(filename).stem,
            "audio_filename": filename,
            "audio_path": audio_path_str,
            "duration_sec": None,
            "sample_rate": None,
            "num_samples": None,
        }
        for key, value in record.items():
            if key == "id":
                continue
            row[key] = value
        rows.append(row)
    audio_stats_map = probe_audio_many(
        audio_paths_to_probe,
        cache_path=audio_cache_path,
        max_workers=audio_probe_workers,
        desc=f"QualiSpeech {split_name} audio",
    )
    for row in rows:
        audio_stats = audio_stats_map.get(row["audio_path"], {"duration_sec": None, "sample_rate": None, "num_samples": None})
        row["duration_sec"] = audio_stats["duration_sec"]
        row["sample_rate"] = audio_stats["sample_rate"]
        row["num_samples"] = audio_stats["num_samples"]
    parsed = pd.DataFrame(rows)
    report = {
        "split": split_name,
        "num_rows": len(parsed),
        "missing_audio_paths": sorted(set(missing_audio_paths)),
        "columns": list(parsed.columns),
    }
    return parsed, report


def parse_qualispeech_dataset(
    root_dir: str | Path,
    output_dir: str | Path,
    *,
    audio_cache_path: str | Path | None = None,
    audio_probe_workers: int | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    split_reports: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    for split_name in tqdm(
        REQUIRED_SPLITS,
        desc="QualiSpeech splits",
        unit="split",
    ):
        parsed, report = parse_qualispeech_split(
            root_dir,
            split_name,
            audio_cache_path=audio_cache_path,
            audio_probe_workers=audio_probe_workers,
        )
        split_reports.append(report)
        write_parquet(output_path / f"qs_{split_name}.parquet", parsed)
        score_columns = [
            "speed",
            "naturalness",
            "background_noise",
            "distortion",
            "listening_effort",
            "continuity",
            "overall_quality",
        ]
        stats = {"split_original": split_name, "num_rows": len(parsed)}
        for column in score_columns:
            if column in parsed.columns:
                stats[f"{column}_mean"] = float(pd.to_numeric(parsed[column], errors="coerce").mean())
        stats_rows.append(stats)
    write_csv(output_path / "qs_stats.csv", pd.DataFrame(stats_rows))
    integrity_report = {"splits": split_reports}
    write_json(output_path / "qs_integrity_report.json", integrity_report)
    return integrity_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse QualiSpeech CSV/audio directories into parquet manifests.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    parse_qualispeech_dataset(args.root_dir, args.output_dir)


if __name__ == "__main__":
    main()
