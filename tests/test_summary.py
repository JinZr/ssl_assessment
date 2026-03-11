from __future__ import annotations

import json

import pandas as pd
import yaml

from src.analysis.summarize import summarize_runs


def _write_run(run_dir, method: str, predictions: pd.DataFrame, variant: str | None = None, reviewer_control: str | None = None) -> None:
    run_dir.mkdir(parents=True)
    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "experiment": {
                    "protocol": "smoke",
                    "encoder": "wavlm_base",
                    "method": method,
                    "variant": variant,
                    "reviewer_control": reviewer_control,
                    "sap_target": "naturalness",
                    "qs_aux": "naturalness" if method != "baseline" else None,
                    "pair_id": "qs_nat_to_sap_nat" if method != "baseline" else None,
                    "ratio": 1.0,
                    "seed": 13,
                    "split_protocol": "paper_faithful",
                }
            },
            handle,
        )
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"mse": 1.0 if method == "baseline" else 0.5, "lcc": 0.5, "srcc": 0.5, "mae": 1.0, "ccc": 0.5, "rmse": 1.0}, handle)


def test_summarize_runs_exports_expected_csvs(tmp_path) -> None:
    runs_dir = tmp_path / "results" / "runs"
    baseline_predictions = pd.DataFrame(
        [
            {"utt_id": "a", "y_true": 1.0, "y_pred": 1.5, "y_pred_clipped": 1.5},
            {"utt_id": "b", "y_true": 2.0, "y_pred": 2.5, "y_pred_clipped": 2.5},
        ]
    )
    jt_predictions = pd.DataFrame(
        [
            {"utt_id": "a", "y_true": 1.0, "y_pred": 1.1, "y_pred_clipped": 1.1},
            {"utt_id": "b", "y_true": 2.0, "y_pred": 2.1, "y_pred_clipped": 2.1},
        ]
    )
    _write_run(runs_dir / "baseline", "baseline", baseline_predictions)
    _write_run(runs_dir / "jt", "jt", jt_predictions)
    _write_run(runs_dir / "ft_head_reset", "ft", jt_predictions, variant="ft_head_reset_reset_full_head", reviewer_control="stage2_head_reset")

    outputs = summarize_runs(runs_dir, tmp_path / "results" / "summaries")
    assert (tmp_path / "results" / "summaries" / "all_results_long.csv").exists()
    assert (tmp_path / "results" / "summaries" / "significance_tests.csv").exists()
    reviewer_results = pd.read_csv(tmp_path / "results" / "summaries" / "reviewer_results.csv")
    assert "ft_head_reset_reset_full_head" in reviewer_results["variant"].fillna("").tolist()
    assert "all_results_mean_std" in outputs
