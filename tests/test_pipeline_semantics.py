from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.cli.pipeline import run_suite
from src.data.split_builder import build_sap_task_split
from src.tasks.pair_builder import sample_auxiliary_frame
from src.utils.io import read_json, read_parquet, write_parquet


def test_sample_auxiliary_frame_matches_effective_n(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "utt_id": [f"utt{i}" for i in range(7)],
            "label_raw": [1, 1, 1, 1, 2, 2, 3],
        }
    )
    sampled_small, meta_small = sample_auxiliary_frame(frame, "label_raw", effective_n=5, seed=13)
    sampled_large, meta_large = sample_auxiliary_frame(frame, "label_raw", effective_n=10, seed=13)
    assert len(sampled_small) == 5
    assert meta_small["effective_n"] == 5
    assert len(sampled_large) == 10
    assert meta_large["effective_n"] == 10
    assert meta_large["unique_n"] == len(frame)


def test_build_sap_task_split_respects_requested_sizes_and_writes_indices(tmp_path) -> None:
    processed_dir = tmp_path / "processed" / "sap"
    processed_dir.mkdir(parents=True)
    utterances = []
    labels = []
    for index in range(80):
        split = "train" if index < 60 else "dev"
        speaker_id = f"spk{index // 5}"
        utt_id = f"utt{index}"
        utterances.append(
            {
                "utt_id": utt_id,
                "speaker_id": speaker_id,
                "split_original": split,
                "audio_path": f"{utt_id}.wav",
            }
        )
        labels.append(
            {
                "utt_id": utt_id,
                "speaker_id": speaker_id,
                "split_original": split,
                "naturalness": float((index % 4) + 1),
            }
        )
    write_parquet(processed_dir / "sap_utterances.parquet", pd.DataFrame(utterances))
    write_parquet(processed_dir / "sap_labels_wide.parquet", pd.DataFrame(labels))

    metadata = build_sap_task_split(
        processed_sap_dir=processed_dir,
        output_dir=tmp_path / "splits",
        task_name="sap_naturalness",
        target_dim="naturalness",
        seed=13,
        protocol="paper_faithful",
        paper_train_size=40,
        paper_val_size=10,
    )
    task_dir = tmp_path / "splits" / "sap_naturalness" / "paper_faithful"
    assert metadata["sap_train_n"] == 40
    assert metadata["sap_val_n"] == 10
    split_ids = read_json(task_dir / "split_indices.json")
    assert len(split_ids["train"]) == 40
    assert len(split_ids["val"]) == 10
    assert (task_dir / "split_index.csv").exists()


def test_run_suite_builds_reviewer_variants_without_alias_methods(monkeypatch, tmp_path) -> None:
    collected: list[dict] = []

    def fake_run_experiment(config, _retry_count: int = 0):  # noqa: ANN001
        collected.append(config)
        return f"run-{len(collected)}"

    monkeypatch.setattr("src.cli.pipeline.run_experiment", fake_run_experiment)

    suite_path = tmp_path / "reviewer_suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "suite_name": "reviewer_test",
                "models": ["wavlm_base_plus"],
                "reviewer_models": ["wavlm_base_plus"],
                "tasks": ["sap_naturalness"],
                "pairs": ["qs_nat_to_sap_nat"],
                "methods": ["jt", "ft"],
                "ratios": [1.0],
                "seeds": [13],
                "protocols": ["paper_faithful"],
                "reviewer_controls": [
                    "sap_multi_task",
                    "dual_head_jt",
                    "shuffled_labels",
                    "stage2_head_reset",
                    "freeze_schedule",
                    "speaker_disjoint_val",
                    "negative_pairs",
                    "huber_loss",
                ],
                "reviewer_protocols": ["speaker_disjoint"],
                "negative_pairs": ["qs_cont_to_sap_nat_neg"],
                "loss_controls": ["mse", "huber"],
                "multitask_tasks": ["sap_naturalness", "sap_intelligibility"],
            }
        ),
        encoding="utf-8",
    )

    repo_root = Path("/Users/zrjin/git/ssl_assessment")
    run_suite(repo_root, suite_path)

    assert any(cfg["experiment"]["method"] == "sap_multi_task" for cfg in collected)
    assert all(cfg["experiment"]["method"] != "ft_head_reset_reset_full_head" for cfg in collected)
    assert any(cfg["experiment"].get("variant", "").startswith("ft_head_reset_") for cfg in collected)
    assert any(cfg["experiment"].get("reviewer_control") == "speaker_disjoint_val" for cfg in collected)
    assert any(cfg["experiment"].get("reviewer_control") == "negative_pairs" for cfg in collected)
    assert any(cfg.get("training", {}).get("loss") == "huber" for cfg in collected)

