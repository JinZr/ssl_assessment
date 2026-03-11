from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src.data.sap_dimension_map import canonicalize_dimension, normalize_dimension_name
from src.utils.audio import probe_audio_many
from src.utils.io import read_json, write_csv, write_json, write_parquet, write_text


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_speaker_jsons(split_dir: Path) -> tuple[list[Path], list[dict[str, str]]]:
    root_jsons = {path.name: path for path in split_dir.glob("*.json")}
    nested_jsons = list(split_dir.glob("*/*.json"))
    duplicates: list[dict[str, str]] = []
    selected: list[Path] = []
    nested_names = {path.name for path in nested_jsons}
    for path in nested_jsons:
        selected.append(path)
        root_match = root_jsons.get(path.name)
        if not root_match:
            continue
        if _file_hash(root_match) != _file_hash(path):
            raise ValueError(f"Conflicting duplicate JSON files: {root_match} vs {path}")
        duplicates.append({"split": split_dir.name, "root_json": str(root_match), "speaker_json": str(path)})
    for name, path in root_jsons.items():
        if name not in nested_names:
            selected.append(path)
    return sorted(selected), duplicates


def _resolve_audio_path(split_dir: Path, speaker_json_path: Path, filename: str) -> Path:
    candidate_paths = [
        speaker_json_path.parent / filename,
        split_dir / filename,
        split_dir / speaker_json_path.stem / filename,
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return candidate_paths[0]


def _extract_prompt_fields(file_entry: dict[str, Any]) -> dict[str, str]:
    prompt = file_entry.get("Prompt") or {}
    return {
        "prompt_text": prompt.get("Prompt Text", "") or "",
        "transcript": prompt.get("Transcript", "") or "",
        "prompt_category": prompt.get("Category Description", "") or "",
        "prompt_subcategory": prompt.get("Sub Category Description", "") or "",
    }


def _parse_rating_level(raw_level: Any) -> float | None:
    if raw_level is None:
        return None
    text = str(raw_level).strip()
    if not text:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or value < 1.0 or value > 7.0:
        return None
    return value


def parse_sap_split(
    split_dir: str | Path,
    *,
    audio_cache_path: str | Path | None = None,
    audio_probe_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_path = Path(split_dir)
    utterance_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    unknown_dimensions: set[str] = set()
    duplicates: list[dict[str, str]] = []
    missing_audio_paths: list[str] = []
    discarded_rating_count = 0
    discarded_examples: list[dict[str, Any]] = []
    json_paths, duplicates = _resolve_speaker_jsons(split_path)
    audio_paths_to_probe: list[str] = []

    for speaker_json_path in tqdm(
        json_paths,
        desc=f"SAP {split_path.name} speakers",
        unit="speaker",
    ):
        payload = read_json(speaker_json_path)
        speaker_id = payload.get("Contributor ID") or speaker_json_path.stem
        etiology = payload.get("Etiology", "") or ""
        block_number = payload.get("BlockNumber")
        files = payload.get("Files") or []
        for file_entry in files:
            filename = file_entry.get("Filename")
            if not filename:
                continue
            audio_path = _resolve_audio_path(split_path, speaker_json_path, filename)
            audio_path_str = str(audio_path)
            if not audio_path.exists():
                missing_audio_paths.append(audio_path_str)
            else:
                audio_paths_to_probe.append(audio_path_str)
            utt_id = Path(filename).stem
            prompt_fields = _extract_prompt_fields(file_entry)
            ratings = file_entry.get("Ratings") or []
            utterance_rows.append(
                {
                    "dataset": "sap",
                    "split_original": split_path.name,
                    "speaker_id": speaker_id,
                    "speaker_dir": speaker_json_path.parent.name,
                    "speaker_json_path": str(speaker_json_path),
                    "utt_id": utt_id,
                    "audio_filename": filename,
                    "audio_path": audio_path_str,
                    "etiology": etiology,
                    "block_number": block_number,
                    "created": file_entry.get("Created"),
                    "created_or_modified": file_entry.get("CreatedOrModified"),
                    "comment": file_entry.get("Comment", "") or "",
                    "prompt_text": prompt_fields["prompt_text"],
                    "transcript": prompt_fields["transcript"],
                    "prompt_category": prompt_fields["prompt_category"],
                    "prompt_subcategory": prompt_fields["prompt_subcategory"],
                    "num_ratings": len(ratings),
                    "has_any_rating": bool(ratings),
                    "duration_sec": None,
                    "sample_rate": None,
                    "num_samples": None,
                }
            )
            for rating in ratings:
                parsed_level = _parse_rating_level(rating.get("Level"))
                if parsed_level is None:
                    discarded_rating_count += 1
                    if len(discarded_examples) < 20:
                        discarded_examples.append(
                            {
                                "utt_id": utt_id,
                                "dimension": normalize_dimension_name(rating.get("Dimension Description", "") or ""),
                                "raw_level": rating.get("Level"),
                            }
                        )
                    continue
                raw_dimension = rating.get("Dimension Description", "") or ""
                canonical_dimension, is_known = canonicalize_dimension(raw_dimension)
                if not is_known:
                    unknown_dimensions.add(normalize_dimension_name(raw_dimension))
                label_rows.append(
                    {
                        "utt_id": utt_id,
                        "speaker_id": speaker_id,
                        "split_original": split_path.name,
                        "dimension_raw": normalize_dimension_name(raw_dimension),
                        "dimension_canonical": canonical_dimension,
                        "label": parsed_level,
                        "label_min": 1.0,
                        "label_max": 7.0,
                    }
                )
    audio_stats_map = probe_audio_many(
        audio_paths_to_probe,
        cache_path=audio_cache_path,
        max_workers=audio_probe_workers,
        desc=f"SAP {split_path.name} audio",
    )
    for row in utterance_rows:
        audio_stats = audio_stats_map.get(row["audio_path"], {"duration_sec": None, "sample_rate": None, "num_samples": None})
        row["duration_sec"] = audio_stats["duration_sec"]
        row["sample_rate"] = audio_stats["sample_rate"]
        row["num_samples"] = audio_stats["num_samples"]
    report = {
        "split": split_path.name,
        "num_speaker_jsons": len(json_paths),
        "num_utterances": len(utterance_rows),
        "num_labels": len(label_rows),
        "num_duplicates": len(duplicates),
        "duplicates": duplicates,
        "unknown_dimensions": sorted(unknown_dimensions),
        "missing_audio_paths": sorted(set(missing_audio_paths)),
        "num_discarded_ratings": discarded_rating_count,
        "discarded_rating_examples": discarded_examples,
    }
    return pd.DataFrame(utterance_rows), pd.DataFrame(label_rows), report


def parse_sap_dataset(
    train_dir: str | Path,
    dev_dir: str | Path,
    output_dir: str | Path,
    *,
    audio_cache_path: str | Path | None = None,
    audio_probe_workers: int | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    all_utterances: list[pd.DataFrame] = []
    all_labels: list[pd.DataFrame] = []
    split_reports: list[dict[str, Any]] = []
    for split_dir in tqdm(
        [train_dir, dev_dir],
        desc="SAP splits",
        unit="split",
    ):
        utterances, labels, report = parse_sap_split(
            split_dir,
            audio_cache_path=audio_cache_path,
            audio_probe_workers=audio_probe_workers,
        )
        all_utterances.append(utterances)
        all_labels.append(labels)
        split_reports.append(report)
    utterance_frame = pd.concat(all_utterances, ignore_index=True)
    label_frame = pd.concat(all_labels, ignore_index=True)
    wide_frame = (
        label_frame.pivot_table(
            index=["utt_id", "speaker_id", "split_original"],
            columns="dimension_canonical",
            values="label",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    dimension_stats = (
        label_frame.groupby(["split_original", "dimension_canonical"], dropna=False)
        .agg(num_labels=("label", "size"), mean_label=("label", "mean"))
        .reset_index()
        .sort_values(["split_original", "dimension_canonical"])
    )
    output_path.mkdir(parents=True, exist_ok=True)
    write_parquet(output_path / "sap_utterances.parquet", utterance_frame)
    write_parquet(output_path / "sap_labels_long.parquet", label_frame)
    write_parquet(output_path / "sap_labels_wide.parquet", wide_frame)
    write_csv(output_path / "sap_dimension_stats.csv", dimension_stats)
    unknown_dimensions = sorted(
        {
            item
            for report in split_reports
            for item in report.get("unknown_dimensions", [])
        }
    )
    write_text(output_path / "unknown_dimensions.log", "\n".join(unknown_dimensions) + ("\n" if unknown_dimensions else ""))
    integrity_report = {
        "splits": split_reports,
        "num_utterances": int(len(utterance_frame)),
        "num_labels": int(len(label_frame)),
        "num_dimensions": int(label_frame["dimension_canonical"].nunique()),
    }
    write_json(output_path / "sap_integrity_report.json", integrity_report)
    return integrity_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse SAP train/dev directories into parquet manifests.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--dev-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    parse_sap_dataset(args.train_dir, args.dev_dir, args.output_dir)


if __name__ == "__main__":
    main()
