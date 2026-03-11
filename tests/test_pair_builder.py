from __future__ import annotations

import pandas as pd

from src.tasks.pair_builder import build_pair_manifests
from src.utils.io import read_parquet, write_parquet


def test_build_pair_manifests_creates_aligned_labels(tmp_path) -> None:
    sap_dir = tmp_path / "splits" / "sap_naturalness" / "paper_faithful"
    qs_dir = tmp_path / "qs"
    sap_dir.mkdir(parents=True)
    qs_dir.mkdir(parents=True)

    sap_train = pd.DataFrame(
        [{"utt_id": "a", "speaker_id": "s1", "split_original": "train", "label": 4.0, "label_min": 1.0, "label_max": 7.0}]
    )
    sap_val = pd.DataFrame(
        [{"utt_id": "b", "speaker_id": "s2", "split_original": "train", "label": 3.0, "label_min": 1.0, "label_max": 7.0}]
    )
    sap_test = pd.DataFrame(
        [{"utt_id": "c", "speaker_id": "s3", "split_original": "dev", "label": 2.0, "label_min": 1.0, "label_max": 7.0}]
    )
    for name, frame in {
        "sap_train_task.parquet": sap_train,
        "sap_val_task.parquet": sap_val,
        "sap_test_task.parquet": sap_test,
    }.items():
        write_parquet(sap_dir / name, frame)

    qs_train = pd.DataFrame([{"utt_id": "q1", "naturalness": 5.0, "audio_path": "q1.wav"}])
    qs_val = pd.DataFrame([{"utt_id": "q2", "naturalness": 1.0, "audio_path": "q2.wav"}])
    write_parquet(qs_dir / "qs_train.parquet", qs_train)
    write_parquet(qs_dir / "qs_val.parquet", qs_val)

    metadata = build_pair_manifests(
        processed_sap_split_dir=sap_dir,
        processed_qs_dir=qs_dir,
        output_dir=tmp_path / "pairs",
        pair_id="qs_nat_to_sap_nat",
        sap_target_dim="naturalness",
        qs_aux_dim="naturalness",
    )

    train_aux = read_parquet(tmp_path / "pairs" / "qs_nat_to_sap_nat" / "paper_faithful" / "qs_train_aux.parquet")
    assert metadata["pair_id"] == "qs_nat_to_sap_nat"
    assert train_aux.loc[0, "label_aligned"] == 1.0
