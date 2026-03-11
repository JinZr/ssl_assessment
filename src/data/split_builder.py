from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import read_parquet, write_csv, write_json, write_parquet
from src.utils.sampling import sample_stratified_exact, select_speaker_subset


def _resolve_val_size(task_name: str, train_n: int, paper_val_size: int | None = None) -> int:
    if paper_val_size is not None:
        target = paper_val_size
    elif task_name in {"sap_naturalness", "sap_intelligibility"}:
        target = 500
    elif train_n >= 500:
        target = 500
    else:
        target = round(0.1 * train_n)
    target = max(1, min(train_n - 1, target))
    return target


def _sample_label_stratified(frame: pd.DataFrame, label_column: str, val_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_frame = sample_stratified_exact(frame, label_column, val_size, seed, replace=False)
    val_keys = set(zip(val_frame["utt_id"].astype(str), val_frame["speaker_id"].astype(str)))
    train_frame = frame[
        ~frame.apply(lambda row: (str(row["utt_id"]), str(row["speaker_id"])) in val_keys, axis=1)
    ].copy()
    if val_frame.empty or train_frame.empty:
        raise ValueError("Stratified split produced an empty train or validation partition.")
    return train_frame.reset_index(drop=True), val_frame.reset_index(drop=True)


def _speaker_disjoint_split(frame: pd.DataFrame, label_column: str, val_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = select_speaker_subset(
        frame=frame,
        speaker_column="speaker_id",
        label_column=label_column,
        target_utterances=val_size,
        seed=seed,
    )
    val_frame = frame[frame["speaker_id"].astype(str).isin(selected.speaker_ids)].copy()
    train_frame = frame[~frame["speaker_id"].astype(str).isin(selected.speaker_ids)].copy()
    if train_frame.empty or val_frame.empty:
        return _sample_label_stratified(frame, label_column, val_size, seed)
    return train_frame.reset_index(drop=True), val_frame.reset_index(drop=True)


def _subsample_paper_train_candidates(
    frame: pd.DataFrame,
    label_column: str,
    paper_train_size: int | None,
    paper_val_size: int,
    seed: int,
) -> pd.DataFrame:
    if paper_train_size is None:
        return frame
    requested_total = paper_train_size + paper_val_size
    if requested_total > len(frame):
        raise ValueError(
            f"Requested paper-faithful train+val size {requested_total} exceeds available rows {len(frame)}."
        )
    return sample_stratified_exact(frame, label_column, requested_total, seed, replace=False)


def build_sap_task_split(
    processed_sap_dir: str | Path,
    output_dir: str | Path,
    task_name: str,
    target_dim: str,
    seed: int = 13,
    protocol: str = "paper_faithful",
    paper_train_size: int | None = None,
    paper_val_size: int | None = None,
) -> dict[str, Any]:
    processed_path = Path(processed_sap_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    utterances = read_parquet(processed_path / "sap_utterances.parquet")
    labels_wide = read_parquet(processed_path / "sap_labels_wide.parquet")
    merged = utterances.merge(labels_wide, on=["utt_id", "speaker_id", "split_original"], how="left")
    if target_dim not in merged.columns:
        raise ValueError(f"Target dimension '{target_dim}' not found in SAP labels.")
    eligible = merged[merged[target_dim].notna()].copy()
    train_candidates = eligible[eligible["split_original"] == "train"].copy()
    test_frame = eligible[eligible["split_original"] == "dev"].copy()
    val_size = _resolve_val_size(task_name, len(train_candidates), paper_val_size=paper_val_size)
    train_candidates = _subsample_paper_train_candidates(
        train_candidates,
        target_dim,
        paper_train_size=paper_train_size,
        paper_val_size=val_size,
        seed=seed,
    )
    if protocol == "speaker_disjoint":
        train_frame, val_frame = _speaker_disjoint_split(train_candidates, target_dim, val_size, seed)
    else:
        train_frame, val_frame = _sample_label_stratified(train_candidates, target_dim, val_size, seed)
    for frame in [train_frame, val_frame, test_frame]:
        frame["target_dim"] = target_dim
        frame["label"] = frame[target_dim]
        frame["label_min"] = 1.0
        frame["label_max"] = 7.0
    task_dir = output_path / task_name / protocol
    write_parquet(task_dir / "sap_train_task.parquet", train_frame)
    write_parquet(task_dir / "sap_val_task.parquet", val_frame)
    write_parquet(task_dir / "sap_test_task.parquet", test_frame)
    split_index = pd.concat(
        [
            train_frame[["utt_id", "speaker_id"]].assign(split="train"),
            val_frame[["utt_id", "speaker_id"]].assign(split="val"),
            test_frame[["utt_id", "speaker_id"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    write_csv(task_dir / "split_index.csv", split_index)
    metadata = {
        "task_name": task_name,
        "target_dim": target_dim,
        "protocol": protocol,
        "random_seed": seed,
        "sap_train_n": int(len(train_frame)),
        "sap_val_n": int(len(val_frame)),
        "sap_test_n": int(len(test_frame)),
        "val_size_requested": int(val_size),
        "paper_train_size_requested": paper_train_size,
        "train_label_histogram": train_frame[target_dim].value_counts(dropna=False).sort_index().to_dict(),
        "val_label_histogram": val_frame[target_dim].value_counts(dropna=False).sort_index().to_dict(),
        "test_label_histogram": test_frame[target_dim].value_counts(dropna=False).sort_index().to_dict(),
    }
    if protocol == "speaker_disjoint":
        metadata["val_speakers"] = sorted(val_frame["speaker_id"].astype(str).unique().tolist())
        metadata["speaker_overlap"] = sorted(
            set(train_frame["speaker_id"].astype(str).unique()).intersection(val_frame["speaker_id"].astype(str).unique())
        )
    write_json(
        task_dir / "split_indices.json",
        {
            "train": sorted(train_frame["utt_id"].astype(str).tolist()),
            "val": sorted(val_frame["utt_id"].astype(str).tolist()),
            "test": sorted(test_frame["utt_id"].astype(str).tolist()),
        },
    )
    write_json(task_dir / "task_metadata.json", metadata)
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SAP task splits from parsed manifests.")
    parser.add_argument("--processed-sap-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--target-dim", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--protocol", default="paper_faithful")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_sap_task_split(
        processed_sap_dir=args.processed_sap_dir,
        output_dir=args.output_dir,
        task_name=args.task_name,
        target_dim=args.target_dim,
        seed=args.seed,
        protocol=args.protocol,
    )


if __name__ == "__main__":
    main()
