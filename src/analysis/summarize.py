from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.breakdowns import export_run_breakdowns
from src.utils.config import load_yaml
from src.utils.io import read_json, write_csv
from src.utils.metrics import lcc, paired_permutation_test, srcc


def _load_run_row(run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "config_resolved.yaml"
    metrics_path = run_dir / "test_metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        return None
    config = load_yaml(config_path)
    metrics = read_json(metrics_path)
    experiment = config.get("experiment", {})
    row = {
        "run_id": run_dir.name,
        "protocol": experiment.get("protocol", "main"),
        "encoder": experiment.get("encoder"),
        "method": experiment.get("method"),
        "variant": experiment.get("variant"),
        "reviewer_control": experiment.get("reviewer_control"),
        "sap_target": experiment.get("sap_target"),
        "qs_aux": experiment.get("qs_aux"),
        "pair_id": experiment.get("pair_id"),
        "ratio": experiment.get("ratio", 1.0),
        "seed": experiment.get("seed"),
        "split_protocol": experiment.get("split_protocol", "paper_faithful"),
    }
    row.update(metrics)
    row["prediction_path"] = str(run_dir / "test_predictions.csv")
    row["config_path"] = str(config_path)
    return row


def _mean_std_table(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "protocol",
        "encoder",
        "method",
        "variant",
        "reviewer_control",
        "sap_target",
        "qs_aux",
        "pair_id",
        "ratio",
        "split_protocol",
    ]
    metric_columns = [column for column in frame.columns if column in {"mse", "lcc", "srcc", "mae", "ccc", "rmse"}]
    grouped = frame.groupby(group_columns, dropna=False)[metric_columns]
    mean_frame = grouped.mean().add_suffix("_mean")
    std_frame = grouped.std(ddof=0).fillna(0.0).add_suffix("_std")
    counts = frame.groupby(group_columns, dropna=False).size().rename("num_seeds")
    merged = pd.concat([mean_frame, std_frame, counts], axis=1).reset_index()
    baseline = merged[(merged["method"] == "baseline") & (merged["variant"].isna())][
        ["encoder", "sap_target", "split_protocol", "mse_mean", "lcc_mean", "srcc_mean"]
    ].rename(
        columns={
            "mse_mean": "baseline_mse_mean",
            "lcc_mean": "baseline_lcc_mean",
            "srcc_mean": "baseline_srcc_mean",
        }
    )
    merged = merged.merge(baseline, on=["encoder", "sap_target", "split_protocol"], how="left")
    merged["mse_gain_mean"] = merged["baseline_mse_mean"] - merged["mse_mean"]
    merged["lcc_gain_mean"] = merged["lcc_mean"] - merged["baseline_lcc_mean"]
    merged["srcc_gain_mean"] = merged["srcc_mean"] - merged["baseline_srcc_mean"]
    return merged


def _best_rows(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    sorted_frame = frame.sort_values(["mse_mean", "lcc_mean", "srcc_mean"], ascending=[True, False, False])
    return sorted_frame.groupby(group_columns, dropna=False, as_index=False).head(1).reset_index(drop=True)


def _paired_metric_difference_ci(
    y_true: np.ndarray,
    run_pred: np.ndarray,
    baseline_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    observed = float(metric_fn(y_true, run_pred) - metric_fn(y_true, baseline_pred))
    rng = np.random.default_rng(seed)
    diffs = np.zeros(n_bootstrap, dtype=np.float64)
    n = len(y_true)
    for index in range(n_bootstrap):
        sample_ids = rng.integers(0, n, size=n)
        truth = y_true[sample_ids]
        diffs[index] = metric_fn(truth, run_pred[sample_ids]) - metric_fn(truth, baseline_pred[sample_ids])
    return observed, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def _significance_rows(frame: pd.DataFrame) -> pd.DataFrame:
    baseline_lookup = {
        (row.encoder, row.sap_target, row.seed, row.split_protocol): Path(row.prediction_path)
        for row in frame.itertuples()
        if row.method == "baseline" and pd.isna(row.variant)
    }
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples():
        if row.method == "baseline":
            continue
        baseline_path = baseline_lookup.get((row.encoder, row.sap_target, row.seed, row.split_protocol))
        if baseline_path is None or not baseline_path.exists():
            continue
        run_predictions = pd.read_csv(row.prediction_path).sort_values("utt_id")
        baseline_predictions = pd.read_csv(baseline_path).sort_values("utt_id")
        merged = run_predictions.merge(
            baseline_predictions[["utt_id", "y_pred"]],
            on="utt_id",
            suffixes=("", "_baseline"),
        )
        if merged.empty:
            continue
        y_true = merged["y_true"].to_numpy(dtype=np.float64)
        y_pred = merged["y_pred"].to_numpy(dtype=np.float64)
        y_pred_baseline = merged["y_pred_baseline"].to_numpy(dtype=np.float64)
        mse_p = paired_permutation_test(y_true, y_pred, y_pred_baseline)
        lcc_delta, lcc_ci_low, lcc_ci_high = _paired_metric_difference_ci(
            y_true,
            y_pred,
            y_pred_baseline,
            lcc,
            n_bootstrap=10_000,
            seed=int(row.seed),
        )
        srcc_delta, srcc_ci_low, srcc_ci_high = _paired_metric_difference_ci(
            y_true,
            y_pred,
            y_pred_baseline,
            srcc,
            n_bootstrap=10_000,
            seed=int(row.seed),
        )
        rows.append(
            {
                "run_id": row.run_id,
                "baseline_prediction_path": str(baseline_path),
                "mse_permutation_p": mse_p,
                "lcc_delta": lcc_delta,
                "lcc_delta_ci_low": lcc_ci_low,
                "lcc_delta_ci_high": lcc_ci_high,
                "srcc_delta": srcc_delta,
                "srcc_delta_ci_low": srcc_ci_low,
                "srcc_delta_ci_high": srcc_ci_high,
            }
        )
    return pd.DataFrame(rows)


def summarize_runs(results_runs_dir: str | Path, summary_dir: str | Path) -> dict[str, str]:
    results_path = Path(results_runs_dir)
    summary_path = Path(summary_dir)
    summary_path.mkdir(parents=True, exist_ok=True)
    rows = [row for run_dir in sorted(results_path.iterdir()) if run_dir.is_dir() for row in [_load_run_row(run_dir)] if row]
    if not rows:
        raise ValueError(f"No completed runs found under {results_path}")
    long_frame = pd.DataFrame(rows).sort_values(["protocol", "encoder", "method", "pair_id", "ratio", "seed"])
    main_frame = long_frame[long_frame["reviewer_control"].isna()].copy()
    mean_std = _mean_std_table(long_frame)
    main_mean_std = _mean_std_table(main_frame) if not main_frame.empty else mean_std.iloc[0:0].copy()
    best_per_encoder = _best_rows(main_mean_std, ["encoder"])
    best_per_pair = _best_rows(main_mean_std[main_mean_std["pair_id"].notna()], ["pair_id"])
    best_per_target = _best_rows(main_mean_std, ["sap_target"])
    ratio_curves = main_mean_std[main_mean_std["method"].isin(["jt", "ft"])].copy()
    reviewer_results = mean_std[
        mean_std["reviewer_control"].notna() | mean_std["variant"].notna()
    ].copy()
    significance = _significance_rows(long_frame)

    write_csv(summary_path / "all_results_long.csv", long_frame)
    write_csv(summary_path / "all_results_mean_std.csv", mean_std)
    write_csv(summary_path / "all_results_with_baseline.csv", mean_std)
    write_csv(summary_path / "best_per_encoder.csv", best_per_encoder)
    write_csv(summary_path / "best_per_pair.csv", best_per_pair)
    write_csv(summary_path / "best_per_target.csv", best_per_target)
    write_csv(summary_path / "ratio_curves.csv", ratio_curves)
    write_csv(summary_path / "reviewer_results.csv", reviewer_results)
    write_csv(summary_path / "significance_tests.csv", significance)

    breakdown_dir = summary_path / "breakdowns"
    breakdown_rows: list[dict[str, str]] = []
    for row in long_frame.itertuples():
        prediction_path = Path(row.prediction_path)
        if prediction_path.exists():
            outputs = export_run_breakdowns(prediction_path, breakdown_dir / row.run_id)
            breakdown_rows.append(
                {
                    "run_id": row.run_id,
                    "prediction_path": str(prediction_path),
                    "prompt_path": outputs.get("prompt", ""),
                    "severity_path": outputs.get("severity", ""),
                    "etiology_path": outputs.get("etiology", ""),
                }
            )
    write_csv(summary_path / "breakdown_index.csv", pd.DataFrame(breakdown_rows))

    return {
        "all_results_long": str(summary_path / "all_results_long.csv"),
        "all_results_mean_std": str(summary_path / "all_results_mean_std.csv"),
        "all_results_with_baseline": str(summary_path / "all_results_with_baseline.csv"),
        "best_per_encoder": str(summary_path / "best_per_encoder.csv"),
        "best_per_pair": str(summary_path / "best_per_pair.csv"),
        "best_per_target": str(summary_path / "best_per_target.csv"),
        "ratio_curves": str(summary_path / "ratio_curves.csv"),
        "reviewer_results": str(summary_path / "reviewer_results.csv"),
        "breakdown_index": str(summary_path / "breakdown_index.csv"),
        "significance_tests": str(summary_path / "significance_tests.csv"),
    }
