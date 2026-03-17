from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import traceback

import pandas as pd
import yaml

from src.cli.pipeline import run_suite
from src.utils.cli import format_compact_exception
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


def test_build_sap_task_split_falls_back_when_paper_request_exceeds_available(tmp_path) -> None:
    processed_dir = tmp_path / "processed" / "sap"
    processed_dir.mkdir(parents=True)
    utterances = []
    labels = []
    for index in range(30):
        split = "train" if index < 24 else "dev"
        speaker_id = f"spk{index // 3}"
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
                "intelligibility": float((index % 4) + 1),
            }
        )
    write_parquet(processed_dir / "sap_utterances.parquet", pd.DataFrame(utterances))
    write_parquet(processed_dir / "sap_labels_wide.parquet", pd.DataFrame(labels))

    metadata = build_sap_task_split(
        processed_sap_dir=processed_dir,
        output_dir=tmp_path / "splits",
        task_name="sap_intelligibility",
        target_dim="intelligibility",
        seed=13,
        protocol="paper_faithful",
        paper_train_size=5046,
        paper_val_size=500,
    )

    assert metadata["paper_size_mode"] == "fallback_to_available"
    assert metadata["paper_total_requested"] == 5546
    assert metadata["paper_total_used"] == 24
    assert metadata["sap_train_n"] + metadata["sap_val_n"] == 24
    assert metadata["sap_val_n"] == metadata["paper_val_size_used"]


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


def test_run_suite_script_honors_suite_lifecycle_flags(monkeypatch, tmp_path) -> None:
    repo_root = Path("/Users/zrjin/git/ssl_assessment")
    suite_path = repo_root / "configs" / "suite" / "tmp_script_suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "prepare": True,
                "summarize": True,
                "tables": False,
                "figures": False,
                "report": False,
                "models": [],
                "tasks": [],
                "pairs": [],
                "methods": [],
                "reviewer_controls": [],
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr("src.cli.pipeline.prepare_all", lambda *args, **kwargs: calls.append(("prepare", None)))
    monkeypatch.setattr("src.cli.pipeline.run_suite", lambda *args, **kwargs: calls.append(("suite", None)) or [])
    monkeypatch.setattr("src.cli.pipeline.run_postprocessing", lambda *args, **kwargs: calls.append(("post", kwargs)) or {})
    monkeypatch.setattr("sys.argv", ["run_suite.py", "--suite", "configs/suite/tmp_script_suite.yaml"])

    try:
        runpy.run_path(str(repo_root / "scripts" / "run_suite.py"), run_name="__main__")
    finally:
        suite_path.unlink(missing_ok=True)

    assert [item[0] for item in calls] == ["prepare", "suite", "post"]
    assert calls[-1][1]["run_summarize"] is True
    assert calls[-1][1]["run_tables"] is False


def test_run_experiment_retries_oom_without_recursive_reentry(monkeypatch, tmp_path) -> None:
    from src.cli.pipeline import run_experiment

    attempts = {"run": 0, "cleanup": 0, "empty_cache": 0, "ipc_collect": 0, "gc": 0}
    statuses: list[tuple[str, dict | None]] = []
    dumped_configs: list[dict] = []
    frame = pd.DataFrame({"utt_id": ["utt0"], "label": [1.0]})

    class FakeTrainer:
        def __init__(self, config, run_dir) -> None:  # noqa: ANN001
            self.config = config
            self.run_dir = run_dir

        def run(self, train_frame, val_frame, test_frame) -> None:  # noqa: ANN001
            attempts["run"] += 1
            if attempts["run"] < 3:
                raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")

        def cleanup(self) -> None:
            attempts["cleanup"] += 1

    monkeypatch.setattr("src.cli.pipeline.seed_everything", lambda seed: None)
    monkeypatch.setattr("src.cli.pipeline.build_run_id", lambda experiment: "oom-run")
    monkeypatch.setattr("src.cli.pipeline.run_complete", lambda run_dir: False)
    monkeypatch.setattr("src.cli.pipeline._resolve_and_lock_model_revision", lambda config: config)
    monkeypatch.setattr("src.cli.pipeline._load_task_frames", lambda config: (frame, frame, frame))
    monkeypatch.setattr("src.cli.pipeline.BaselineTrainer", FakeTrainer)
    monkeypatch.setattr(
        "src.cli.pipeline.dump_yaml",
        lambda path, payload: dumped_configs.append(deepcopy(payload)),
    )
    monkeypatch.setattr(
        "src.cli.pipeline.write_run_status",
        lambda run_dir, status, extra=None: statuses.append((status, deepcopy(extra))),
    )
    monkeypatch.setattr("src.cli.pipeline.gc.collect", lambda: attempts.__setitem__("gc", attempts["gc"] + 1) or 0)
    monkeypatch.setattr("src.cli.pipeline.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "src.cli.pipeline.torch.cuda.empty_cache",
        lambda: attempts.__setitem__("empty_cache", attempts["empty_cache"] + 1),
    )
    monkeypatch.setattr(
        "src.cli.pipeline.torch.cuda.ipc_collect",
        lambda: attempts.__setitem__("ipc_collect", attempts["ipc_collect"] + 1),
    )

    config = {
        "experiment": {
            "seed": 13,
            "method": "baseline",
            "encoder": "wavlm_base",
            "split_protocol": "paper_faithful",
            "task_name": "sap_naturalness",
            "oom_max_retries": 3,
            "oom_min_budget_sec": 10,
        },
        "model": {"name": "wavlm_base", "max_total_sec": 180},
        "paths": {"results_dir": str(tmp_path / "results"), "metadata_dir": str(tmp_path / "metadata")},
        "results": {"skip_if_complete": False},
    }

    run_id = run_experiment(config)

    assert run_id == "oom-run"
    assert attempts["run"] == 3
    assert attempts["cleanup"] == 3
    assert attempts["gc"] == 2
    assert attempts["empty_cache"] == 2
    assert attempts["ipc_collect"] == 2
    assert [status for status, _ in statuses] == ["running", "oom_retry", "running", "oom_retry", "running", "complete"]
    assert dumped_configs[-1]["experiment"]["max_total_sec_override"] == 45
    assert dumped_configs[-1]["experiment"]["max_input_sec_override"] == 45


def test_run_suite_applies_suite_level_config_overrides(monkeypatch, tmp_path) -> None:
    collected: list[dict] = []

    def fake_run_experiment(config, _retry_count: int = 0):  # noqa: ANN001
        collected.append(config)
        return f"run-{len(collected)}"

    monkeypatch.setattr("src.cli.pipeline.run_experiment", fake_run_experiment)

    suite_path = tmp_path / "suite_with_overrides.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "suite_name": "override_test",
                "config_overrides": {
                    "model": {
                        "max_total_sec": 45,
                        "max_input_sec": 30,
                    }
                },
                "models": ["wavlm_base"],
                "tasks": ["sap_naturalness"],
                "pairs": [],
                "methods": ["baseline"],
                "seeds": [13],
                "protocols": ["paper_faithful"],
            }
        ),
        encoding="utf-8",
    )

    repo_root = Path("/Users/zrjin/git/ssl_assessment")
    run_suite(repo_root, suite_path)

    assert len(collected) == 1
    assert collected[0]["model"]["max_total_sec"] == 45
    assert collected[0]["model"]["max_input_sec"] == 30


def test_format_compact_exception_filters_site_packages_frames(monkeypatch, tmp_path) -> None:
    error = RuntimeError("CUDA out of memory")
    error.__traceback__ = None
    frames = [
        traceback.FrameSummary(
            filename="/mnt/env/lib/python3.10/site-packages/torch/utils/checkpoint.py",
            lineno=325,
            name="backward",
            line="torch.autograd.backward(outputs_with_grad, args_with_grad)",
        ),
        traceback.FrameSummary(
            filename=str(tmp_path / "src" / "cli" / "pipeline.py"),
            lineno=328,
            name="run_experiment",
            line="write_run_status(run_dir, 'oom_retry', {'max_total_sec': next_budget})",
        ),
    ]
    monkeypatch.setattr("src.utils.cli.traceback.extract_tb", lambda tb: frames)

    rendered = format_compact_exception(error, tmp_path)

    assert "RuntimeError: CUDA out of memory" in rendered
    assert "site-packages" not in rendered
    assert "src/cli/pipeline.py:328 in run_experiment" in rendered
