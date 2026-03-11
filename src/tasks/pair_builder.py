from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import read_parquet, write_json, write_parquet
from src.utils.sampling import sample_stratified_exact


def map_qs_score_to_sap(qs_score: float) -> float:
    return 1.0 + (5.0 - float(qs_score)) * 6.0 / 4.0


def sample_auxiliary_frame(
    frame: pd.DataFrame,
    label_column: str,
    effective_n: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    full_n = len(frame)
    if effective_n <= full_n:
        sampled = sample_stratified_exact(frame, label_column, effective_n, seed, replace=False)
        metadata = {
            "effective_n": int(effective_n),
            "unique_n": int(len(sampled)),
            "oversample_factor": 1.0,
        }
        return sampled, metadata

    sampled_parts = [frame.copy()]
    remaining = effective_n - full_n
    sampled_parts.append(sample_stratified_exact(frame, label_column, remaining, seed + 1, replace=True))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    metadata = {
        "effective_n": int(effective_n),
        "unique_n": int(full_n),
        "oversample_factor": float(effective_n / max(1, full_n)),
    }
    return sampled, metadata


def build_pair_manifests(
    processed_sap_split_dir: str | Path,
    processed_qs_dir: str | Path,
    output_dir: str | Path,
    pair_id: str,
    sap_target_dim: str,
    qs_aux_dim: str,
    random_seed: int = 13,
    split_protocol: str = "paper_faithful",
) -> dict[str, Any]:
    sap_path = Path(processed_sap_split_dir)
    qs_path = Path(processed_qs_dir)
    out_dir = Path(output_dir) / pair_id / split_protocol
    out_dir.mkdir(parents=True, exist_ok=True)

    sap_train = read_parquet(sap_path / "sap_train_task.parquet").copy()
    sap_val = read_parquet(sap_path / "sap_val_task.parquet").copy()
    sap_test = read_parquet(sap_path / "sap_test_task.parquet").copy()
    qs_train = read_parquet(qs_path / "qs_train.parquet").copy()
    qs_val = read_parquet(qs_path / "qs_val.parquet").copy()

    qs_train = qs_train[qs_train[qs_aux_dim].notna()].copy()
    qs_val = qs_val[qs_val[qs_aux_dim].notna()].copy()
    qs_train["aux_dim"] = qs_aux_dim
    qs_val["aux_dim"] = qs_aux_dim

    for frame in [sap_train, sap_val, sap_test]:
        frame["label_for_loss"] = frame["label"]
        frame["domain"] = "sap"
        frame["task_dim"] = sap_target_dim
        frame["label_raw"] = frame["label"]
        frame["label_aligned"] = frame["label"]
    for frame in [qs_train, qs_val]:
        frame["domain"] = "qualispeech"
        frame["task_dim"] = qs_aux_dim
        frame["label_raw"] = pd.to_numeric(frame[qs_aux_dim], errors="coerce")
        frame["label_aligned"] = frame["label_raw"].apply(map_qs_score_to_sap)
        frame["label_min"] = 1.0
        frame["label_max"] = 5.0

    write_parquet(out_dir / "sap_train_task.parquet", sap_train)
    write_parquet(out_dir / "sap_val_task.parquet", sap_val)
    write_parquet(out_dir / "sap_test_task.parquet", sap_test)
    write_parquet(out_dir / "qs_train_aux.parquet", qs_train)
    write_parquet(out_dir / "qs_val_aux.parquet", qs_val)

    metadata = {
        "pair_id": pair_id,
        "sap_target_dim": sap_target_dim,
        "qs_aux_dim": qs_aux_dim,
        "sap_train_n": int(len(sap_train)),
        "sap_val_n": int(len(sap_val)),
        "sap_test_n": int(len(sap_test)),
        "qs_train_n": int(len(qs_train)),
        "qs_val_n": int(len(qs_val)),
        "label_range_sap": [1, 7],
        "label_range_qs": [1, 5],
        "jt_mapping_formula": "1 + (5 - qs_score) * 6 / 4",
        "split_protocol": split_protocol,
        "random_seed": random_seed,
    }
    write_json(out_dir / "pair_metadata.json", metadata)
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build pair-specific SAP and QualiSpeech manifests.")
    parser.add_argument("--sap-split-dir", required=True)
    parser.add_argument("--qs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--sap-target-dim", required=True)
    parser.add_argument("--qs-aux-dim", required=True)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--split-protocol", default="paper_faithful")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_pair_manifests(
        processed_sap_split_dir=args.sap_split_dir,
        processed_qs_dir=args.qs_dir,
        output_dir=args.output_dir,
        pair_id=args.pair_id,
        sap_target_dim=args.sap_target_dim,
        qs_aux_dim=args.qs_aux_dim,
        random_seed=args.random_seed,
        split_protocol=args.split_protocol,
    )


if __name__ == "__main__":
    main()
