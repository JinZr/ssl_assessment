from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.parse_qualispeech import parse_qualispeech_dataset
from src.data.parse_sap import parse_sap_dataset
from src.data.split_builder import build_sap_task_split
from src.tasks.pair_builder import build_pair_manifests
from src.trainers.baseline_trainer import BaselineTrainer
from src.trainers.ft_trainer import FTTrainer
from src.trainers.jt_trainer import JTTrainer
from src.trainers.reviewer_controls import DualHeadJTTrainer, SAPMultiTaskTrainer
from src.utils.config import dump_yaml, load_config_bundle, load_yaml
from src.utils.experiment import build_run_id, run_complete, write_run_status
from src.utils.hf import resolved_revision_record
from src.utils.io import read_json, read_parquet, write_json
from src.models.model_registry import get_model_spec
from src.utils.seed import seed_everything


def _load_task_configs(task_dir: str | Path) -> dict[str, dict[str, Any]]:
    return {path.stem: load_yaml(path) for path in sorted(Path(task_dir).glob("*.yaml"))}


def _load_pair_configs(pair_dir: str | Path) -> dict[str, dict[str, Any]]:
    return {path.stem: load_yaml(path) for path in sorted(Path(pair_dir).glob("*.yaml"))}


def _runs_dir(paths: dict[str, Any]) -> Path:
    results_dir = Path(paths["results_dir"])
    return results_dir if results_dir.name == "runs" else results_dir / "runs"


def _results_root(paths: dict[str, Any]) -> Path:
    runs_dir = _runs_dir(paths)
    return runs_dir.parent


def prepare_all(paths_config: dict[str, Any], task_configs: dict[str, dict[str, Any]], pair_configs: dict[str, dict[str, Any]]) -> None:
    paths = paths_config["paths"]
    default_seed = int(paths_config.get("defaults", {}).get("seed", 13))
    parse_sap_dataset(paths["sap"]["train_dir"], paths["sap"]["dev_dir"], paths["processed"]["sap_dir"])
    parse_qualispeech_dataset(paths["qualispeech"]["root_dir"], paths["processed"]["qs_dir"])

    for task_config in tqdm(
        task_configs.values(),
        desc="Build SAP task splits",
        unit="task",
    ):
        build_sap_task_split(
            processed_sap_dir=paths["processed"]["sap_dir"],
            output_dir=paths["processed"]["splits_dir"],
            task_name=task_config["task_name"],
            target_dim=task_config["target_dim"],
            seed=default_seed,
            protocol="paper_faithful",
            paper_train_size=task_config.get("paper_train_size"),
            paper_val_size=task_config.get("paper_val_size"),
        )
        build_sap_task_split(
            processed_sap_dir=paths["processed"]["sap_dir"],
            output_dir=paths["processed"]["splits_dir"],
            task_name=task_config["task_name"],
            target_dim=task_config["target_dim"],
            seed=default_seed,
            protocol="speaker_disjoint",
            paper_train_size=task_config.get("paper_train_size"),
            paper_val_size=task_config.get("paper_val_size"),
        )

    for pair_config in tqdm(
        pair_configs.values(),
        desc="Build pair manifests",
        unit="pair",
    ):
        split_dir = Path(paths["processed"]["splits_dir"]) / pair_config["sap_target_task"]
        for protocol in ("paper_faithful", "speaker_disjoint"):
            build_pair_manifests(
                processed_sap_split_dir=split_dir / protocol,
                processed_qs_dir=paths["processed"]["qs_dir"],
                output_dir=paths["processed"]["pairs_dir"],
                pair_id=pair_config["pair_id"],
                sap_target_dim=pair_config["sap_target_dim"],
                qs_aux_dim=pair_config["qs_aux_dim"],
                random_seed=default_seed,
                split_protocol=protocol,
            )


def _append_model_revision(metadata_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = config["model"]
    spec = get_model_spec(model_cfg["name"])
    record = resolved_revision_record(model_cfg["name"], spec.model_id, model_cfg.get("revision"))
    target = metadata_dir / "resolved_model_revisions.yaml"
    existing = load_yaml(target) if target.exists() else {"models": []}
    models = [item for item in existing.get("models", []) if item.get("model_name") != record["model_name"]]
    models.append(record)
    dump_yaml(target, {"models": sorted(models, key=lambda item: item["model_name"])})
    return record


def _read_split_ids(task_dir: str | Path) -> dict[str, list[str]]:
    index_path = Path(task_dir) / "split_indices.json"
    if index_path.exists():
        return read_json(index_path)
    return {
        "train": read_parquet(Path(task_dir) / "sap_train_task.parquet")["utt_id"].astype(str).tolist(),
        "val": read_parquet(Path(task_dir) / "sap_val_task.parquet")["utt_id"].astype(str).tolist(),
        "test": read_parquet(Path(task_dir) / "sap_test_task.parquet")["utt_id"].astype(str).tolist(),
    }


def _load_task_frames(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_dir = (
        Path(config["paths"]["processed"]["splits_dir"])
        / config["experiment"]["task_name"]
        / config["experiment"]["split_protocol"]
    )
    return (
        read_parquet(task_dir / "sap_train_task.parquet"),
        read_parquet(task_dir / "sap_val_task.parquet"),
        read_parquet(task_dir / "sap_test_task.parquet"),
    )


def _load_pair_frames(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_dir = (
        Path(config["paths"]["processed"]["pairs_dir"])
        / config["experiment"]["pair_id"]
        / config["experiment"]["split_protocol"]
    )
    return (
        read_parquet(pair_dir / "sap_train_task.parquet"),
        read_parquet(pair_dir / "sap_val_task.parquet"),
        read_parquet(pair_dir / "sap_test_task.parquet"),
        read_parquet(pair_dir / "qs_train_aux.parquet"),
        read_parquet(pair_dir / "qs_val_aux.parquet"),
    )


def _shuffle_aux_labels(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    shuffled = frame.copy()
    rng = np.random.default_rng(seed)
    shuffled_values = shuffled["label_raw"].to_numpy().copy()
    rng.shuffle(shuffled_values)
    shuffled["label_raw"] = shuffled_values
    if "label_aligned" in shuffled.columns:
        aligned_values = shuffled["label_aligned"].to_numpy().copy()
        rng.shuffle(aligned_values)
        shuffled["label_aligned"] = aligned_values
    return shuffled


def _build_multitask_frames(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_protocol = config["experiment"]["split_protocol"]
    eval_task = config["experiment"]["task_name"]
    task_names = list(config["experiment"]["multitask_tasks"])
    target_dims = dict(config["experiment"]["multitask_target_dims"])
    processed_sap_dir = Path(config["paths"]["processed"]["sap_dir"])
    utterances = read_parquet(processed_sap_dir / "sap_utterances.parquet")
    labels_wide = read_parquet(processed_sap_dir / "sap_labels_wide.parquet")
    merged = utterances.merge(labels_wide, on=["utt_id", "speaker_id", "split_original"], how="left")

    eval_task_dir = Path(config["paths"]["processed"]["splits_dir"]) / eval_task / split_protocol
    split_ids = {split_name: set(ids) for split_name, ids in _read_split_ids(eval_task_dir).items()}

    def build_partition(split_name: str) -> pd.DataFrame:
        partition = merged[merged["utt_id"].astype(str).isin(split_ids[split_name])].copy()
        partition["task_labels"] = partition.apply(
            lambda row: {task_name: row.get(target_dims[task_name]) for task_name in task_names},
            axis=1,
        )
        partition["eval_task_id"] = eval_task
        partition["label"] = partition[target_dims[eval_task]]
        partition["label_for_loss"] = partition["label"]
        partition["label_min"] = 1.0
        partition["label_max"] = 7.0
        partition["target_dim"] = target_dims[eval_task]
        partition["domain"] = "sap"
        return partition.reset_index(drop=True)

    return build_partition("train"), build_partition("val"), build_partition("test")


def _apply_loss_override(config: dict[str, Any], loss_name: str, huber_delta: float | None = None) -> None:
    config.setdefault("training", {})["loss"] = loss_name
    if loss_name == "huber" and huber_delta is not None:
        config["training"]["huber_delta"] = huber_delta


def _apply_reviewer_variant(config: dict[str, Any], variant: str, control: str) -> None:
    config.setdefault("experiment", {})["variant"] = variant
    config["experiment"]["reviewer_control"] = control


def _resolve_and_lock_model_revision(config: dict[str, Any]) -> dict[str, Any]:
    spec = get_model_spec(config["model"]["name"])
    config["model"]["model_id"] = spec.model_id
    record = _append_model_revision(Path(config["paths"]["metadata_dir"]), config)
    config["model"]["requested_revision"] = record["requested_revision"]
    config["model"]["resolved_revision"] = record["resolved_revision"]
    return config


def run_experiment(config: dict[str, Any], _retry_count: int = 0) -> str:
    config = deepcopy(config)
    seed_everything(int(config["experiment"]["seed"]))
    run_id = build_run_id(config["experiment"])
    run_dir = _runs_dir(config["paths"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.get("results", {}).get("skip_if_complete", True) and run_complete(run_dir):
        status = load_yaml(run_dir / "run_status.json")
        if status.get("status") == "complete":
            return run_id
    config = _resolve_and_lock_model_revision(config)
    dump_yaml(run_dir / "config_resolved.yaml", config)
    write_run_status(run_dir, "running")
    trainer = None
    try:
        method = config["experiment"]["method"]
        if method == "baseline":
            train_frame, val_frame, test_frame = _load_task_frames(config)
            trainer = BaselineTrainer(config, run_dir)
            trainer.run(train_frame, val_frame, test_frame)
        elif method in {"jt", "jt_shuffled"}:
            sap_train, sap_val, sap_test, qs_train, _ = _load_pair_frames(config)
            if method == "jt_shuffled":
                qs_train = _shuffle_aux_labels(qs_train, int(config["experiment"]["seed"]))
            trainer = JTTrainer(config, run_dir)
            trainer.run(sap_train, sap_val, sap_test, qs_train)
        elif method == "dual_head_jt":
            sap_train, sap_val, sap_test, qs_train, _ = _load_pair_frames(config)
            trainer = DualHeadJTTrainer(config, run_dir)
            trainer.run(sap_train, sap_val, sap_test, qs_train)
        elif method in {"ft", "ft_shuffled"}:
            sap_train, sap_val, sap_test, qs_train, qs_val = _load_pair_frames(config)
            if method == "ft_shuffled":
                qs_train = _shuffle_aux_labels(qs_train, int(config["experiment"]["seed"]))
                qs_val = _shuffle_aux_labels(qs_val, int(config["experiment"]["seed"]) + 1)
            trainer = FTTrainer(config, run_dir)
            trainer.run(qs_train, qs_val, sap_train, sap_val, sap_test)
        elif method == "sap_multi_task":
            train_frame, val_frame, test_frame = _build_multitask_frames(config)
            trainer = SAPMultiTaskTrainer(config, run_dir)
            trainer.run(train_frame, val_frame, test_frame)
        else:
            raise ValueError(f"Unsupported experiment method: {method}")
        write_run_status(run_dir, "complete", {"run_id": run_id})
        return run_id
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            current_budget = config.get("training", {}).get("max_total_sec")
            if current_budget is None:
                current_budget = config["model"].get("max_total_sec", 180)
            next_budget = max(10, int(current_budget // 2))
            if _retry_count >= 1 or next_budget == current_budget:
                write_run_status(
                    run_dir,
                    "failed",
                    {
                        "error": str(error),
                        "failure_reason": "oom_after_retry",
                        "max_total_sec": current_budget,
                    },
                )
                raise
            config.setdefault("experiment", {})["max_total_sec_override"] = next_budget
            dump_yaml(run_dir / "config_resolved.yaml", config)
            write_run_status(run_dir, "oom_retry", {"max_total_sec": next_budget, "retry_count": _retry_count + 1})
            if trainer is not None:
                trainer.cleanup()
                trainer = None
            return run_experiment(config, _retry_count=_retry_count + 1)
        write_run_status(run_dir, "failed", {"error": str(error)})
        raise
    except Exception as error:
        write_run_status(run_dir, "failed", {"error": str(error)})
        raise
    finally:
        if trainer is not None:
            trainer.cleanup()


def _method_config_name(method: str) -> str:
    if method in {"baseline", "jt", "ft"}:
        return method
    return "reviewer_controls"


def _compose_run_config(
    repo_root: Path,
    suite_name: str,
    model_name: str,
    method: str,
    seed: int,
    ratio: float,
    split_protocol: str,
    task_name: str | None = None,
    pair_name: str | None = None,
    extra_experiment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_config_bundle(
        repo_root / "configs" / "defaults.yaml",
        repo_root / "configs" / "paths.yaml",
        extra_paths=[
            repo_root / "configs" / "models" / f"{model_name}.yaml",
            repo_root / "configs" / "experiments" / f"{_method_config_name(method)}.yaml",
        ],
    )
    if task_name:
        config = load_config_bundle(
            repo_root / "configs" / "defaults.yaml",
            repo_root / "configs" / "paths.yaml",
            extra_paths=[
                repo_root / "configs" / "models" / f"{model_name}.yaml",
                repo_root / "configs" / "experiments" / f"{_method_config_name(method)}.yaml",
                repo_root / "configs" / "tasks" / f"{task_name}.yaml",
            ],
        )
    if pair_name:
        extras = [
            repo_root / "configs" / "models" / f"{model_name}.yaml",
            repo_root / "configs" / "experiments" / f"{_method_config_name(method)}.yaml",
            repo_root / "configs" / "pairs" / f"{pair_name}.yaml",
        ]
        config = load_config_bundle(repo_root / "configs" / "defaults.yaml", repo_root / "configs" / "paths.yaml", extra_paths=extras)
    experiment = config.setdefault("experiment", {})
    experiment.update(
        {
            "protocol": suite_name,
            "encoder": model_name,
            "method": method,
            "ratio": ratio,
            "seed": seed,
            "split_protocol": split_protocol,
        }
    )
    if task_name:
        experiment["task_name"] = config["task_name"]
        experiment["sap_target"] = config["target_dim"]
    if pair_name:
        experiment["pair_id"] = config["pair_id"]
        experiment["sap_target"] = config["sap_target_dim"]
        experiment["qs_aux"] = config["qs_aux_dim"]
    if extra_experiment:
        experiment.update(extra_experiment)
    return config


def _build_task_dim_map(repo_root: Path, task_names: list[str]) -> dict[str, str]:
    return {
        task_name: load_yaml(repo_root / "configs" / "tasks" / f"{task_name}.yaml")["target_dim"]
        for task_name in task_names
    }


def run_suite(repo_root: str | Path, suite_config_path: str | Path) -> list[str]:
    root = Path(repo_root)
    suite_config = load_yaml(suite_config_path)
    run_ids: list[str] = []
    if suite_config.get("sub_suites"):
        for relative in suite_config["sub_suites"]:
            run_ids.extend(run_suite(root, root / relative))
        return run_ids

    suite_name = suite_config.get("suite_name", Path(suite_config_path).stem)
    models = suite_config.get("models", [])
    reviewer_models = suite_config.get("reviewer_models", models)
    tasks = suite_config.get("tasks", [])
    pairs = suite_config.get("pairs", [])
    methods = suite_config.get("methods", [])
    seeds = suite_config.get("seeds", [13])
    ratios = suite_config.get("ratios", [1.0])
    split_protocols = suite_config.get("split_protocols") or suite_config.get("protocols") or ["paper_faithful"]
    reviewer_controls = suite_config.get("reviewer_controls", {})
    if isinstance(reviewer_controls, list):
        reviewer_controls = {"controls": reviewer_controls}
    reviewer_cfg = load_yaml(root / "configs" / "experiments" / "reviewer_controls.yaml").get("reviewer", {})
    controls = reviewer_controls.get("controls", reviewer_controls.get("methods", []))
    reviewer_protocols = suite_config.get("reviewer_protocols", reviewer_cfg.get("reviewer_protocols", []))
    negative_pairs = suite_config.get("negative_pairs", reviewer_cfg.get("negative_pairs", []))
    loss_controls = suite_config.get("loss_controls", reviewer_cfg.get("ft", {}).get("loss_options", []))
    multitask_tasks = suite_config.get("multitask_tasks", tasks)
    multitask_target_dims = _build_task_dim_map(root, multitask_tasks)

    def maybe_run(config: dict[str, Any]) -> None:
        run_ids.append(run_experiment(config))

    for split_protocol in split_protocols:
        for model_name in models:
            for seed in seeds:
                for task_name in tasks:
                    if "baseline" in methods:
                        config = _compose_run_config(root, suite_name, model_name, "baseline", seed, 1.0, split_protocol, task_name=task_name)
                        maybe_run(config)
                for pair_name in pairs:
                    for ratio in ratios:
                        for method in methods:
                            if method == "baseline":
                                continue
                            config = _compose_run_config(root, suite_name, model_name, method, seed, ratio, split_protocol, pair_name=pair_name)
                            maybe_run(config)

        for model_name in reviewer_models:
            for seed in seeds:
                if "sap_multi_task" in controls:
                    for task_name in tasks:
                        config = _compose_run_config(
                            root,
                            suite_name,
                            model_name,
                            "sap_multi_task",
                            seed,
                            1.0,
                            split_protocol,
                            task_name=task_name,
                            extra_experiment={
                                "multitask_tasks": multitask_tasks,
                                "multitask_target_dims": multitask_target_dims,
                                "variant": "sap_multi_task",
                            },
                        )
                        maybe_run(config)

                for pair_name in pairs:
                    if "dual_head_jt" in controls:
                        config = _compose_run_config(root, suite_name, model_name, "dual_head_jt", seed, 1.0, split_protocol, pair_name=pair_name)
                        _apply_reviewer_variant(config, "dual_head", "dual_head_jt")
                        maybe_run(config)
                    if "shuffled_labels" in controls:
                        for method in ("jt_shuffled", "ft_shuffled"):
                            config = _compose_run_config(root, suite_name, model_name, method, seed, 1.0, split_protocol, pair_name=pair_name)
                            _apply_reviewer_variant(config, method.replace("_shuffled", "_shuffle"), "shuffled_labels")
                            maybe_run(config)
                    if "stage2_head_reset" in controls:
                        head_reset_options = reviewer_cfg.get("ft", {}).get("head_reset_options", [])
                        for reset_mode in head_reset_options:
                            config = _compose_run_config(root, suite_name, model_name, "ft", seed, 1.0, split_protocol, pair_name=pair_name)
                            config.setdefault("ft", {})["head_reset"] = reset_mode
                            _apply_reviewer_variant(config, f"ft_head_reset_{reset_mode}", "stage2_head_reset")
                            maybe_run(config)
                    if "freeze_schedule" in controls:
                        freeze_schedules = reviewer_cfg.get("ft", {}).get("freeze_schedules", [])
                        for schedule in freeze_schedules:
                            config = _compose_run_config(root, suite_name, model_name, "ft", seed, 1.0, split_protocol, pair_name=pair_name)
                            config.setdefault("ft", {})["freeze_schedule"] = schedule
                            _apply_reviewer_variant(config, f"ft_freeze_{schedule}", "freeze_schedule")
                            maybe_run(config)
                    if "huber_loss" in controls and "huber" in loss_controls:
                        for method in [item for item in methods if item in {"jt", "ft"}]:
                            config = _compose_run_config(root, suite_name, model_name, method, seed, 1.0, split_protocol, pair_name=pair_name)
                            _apply_loss_override(config, "huber", reviewer_cfg.get("ft", {}).get("huber_delta", 1.0))
                            _apply_reviewer_variant(config, f"{method}_loss_huber", "huber_loss")
                            maybe_run(config)
                if "huber_loss" in controls and "baseline" in methods and "huber" in loss_controls:
                    for task_name in tasks:
                        config = _compose_run_config(root, suite_name, model_name, "baseline", seed, 1.0, split_protocol, task_name=task_name)
                        _apply_loss_override(config, "huber", reviewer_cfg.get("ft", {}).get("huber_delta", 1.0))
                        _apply_reviewer_variant(config, "baseline_loss_huber", "huber_loss")
                        maybe_run(config)
                if "negative_pairs" in controls:
                    for pair_name in negative_pairs:
                        for method in [item for item in methods if item in {"jt", "ft"}]:
                            config = _compose_run_config(root, suite_name, model_name, method, seed, 1.0, split_protocol, pair_name=pair_name)
                            _apply_reviewer_variant(config, f"negative_pair_{pair_name}", "negative_pairs")
                            maybe_run(config)

    if "speaker_disjoint_val" in controls:
        for split_protocol in reviewer_protocols:
            for model_name in reviewer_models:
                for seed in seeds:
                    for task_name in tasks:
                        config = _compose_run_config(root, suite_name, model_name, "baseline", seed, 1.0, split_protocol, task_name=task_name)
                        _apply_reviewer_variant(config, "speaker_disjoint_baseline", "speaker_disjoint_val")
                        maybe_run(config)
                    for pair_name in pairs:
                        for method in [item for item in methods if item in {"jt", "ft"}]:
                            config = _compose_run_config(root, suite_name, model_name, method, seed, 1.0, split_protocol, pair_name=pair_name)
                            _apply_reviewer_variant(config, f"speaker_disjoint_{method}", "speaker_disjoint_val")
                            maybe_run(config)
    return run_ids


def run_postprocessing(
    paths_config: dict[str, Any],
    *,
    run_summarize: bool = True,
    run_tables: bool = True,
    run_figures: bool = True,
    run_report: bool = True,
) -> dict[str, str]:
    from src.analysis.reporting import package_markdown_report
    from src.analysis.summarize import summarize_runs
    from src.plots.figures import export_breakdown_figures, export_gain_figures, export_prediction_figures, export_ratio_figures
    from src.tables.export import export_tables

    paths = paths_config["paths"]
    results_root = _results_root(paths)
    runs_dir = _runs_dir(paths)
    summary_outputs: dict[str, str] = {}
    if run_summarize:
        summary_outputs = summarize_runs(runs_dir, results_root / "summaries")
    if run_tables:
        export_tables(results_root / "summaries", results_root / "tables")
    if run_figures:
        export_ratio_figures(results_root / "summaries", results_root / "figures")
        export_gain_figures(results_root / "summaries", results_root / "figures")
        export_prediction_figures(runs_dir, results_root / "figures")
        export_breakdown_figures(results_root / "summaries", results_root / "figures")
    if run_report:
        report_path = package_markdown_report(
            results_root / "summaries",
            results_root / "figures",
            results_root / "reports" / "report.md",
        )
        write_json(results_root / "reports" / "report_manifest.json", {"report_path": report_path, **summary_outputs})
    return summary_outputs
