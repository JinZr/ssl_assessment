from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import write_csv, write_json
from src.utils.metrics import bootstrap_ci, compute_metrics


def build_prediction_frame(
    records: list[dict[str, Any]],
    run_metadata: dict[str, Any],
    clip_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    lower, upper = clip_range or (
        float(frame["label_min"].dropna().min()),
        float(frame["label_max"].dropna().max()),
    )
    frame["y_pred_clipped"] = frame["y_pred"].clip(lower=lower, upper=upper)
    for key, value in run_metadata.items():
        frame[key] = value
    return frame


def metric_payload(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = 13,
) -> dict[str, Any]:
    y_true = frame["y_true"].to_numpy(dtype=np.float64)
    y_pred = frame["y_pred"].to_numpy(dtype=np.float64)
    y_pred_clipped = frame["y_pred_clipped"].to_numpy(dtype=np.float64)
    raw_metrics = compute_metrics(y_true, y_pred).to_dict()
    clipped_metrics = compute_metrics(y_true, y_pred_clipped).to_dict()
    payload: dict[str, Any] = dict(raw_metrics)
    payload.update({f"clipped_{key}": value for key, value in clipped_metrics.items()})
    for metric_name in ("mse", "lcc", "srcc"):
        low, high = bootstrap_ci(
            y_true,
            y_pred,
            lambda truth, pred, name=metric_name: compute_metrics(truth, pred).to_dict()[name],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        payload[f"{metric_name}_ci_low"] = low
        payload[f"{metric_name}_ci_high"] = high
    return payload


def export_evaluation(
    output_dir: str | Path,
    prediction_records: list[dict[str, Any]],
    run_metadata: dict[str, Any],
    n_bootstrap: int = 10_000,
    seed: int = 13,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_frame = build_prediction_frame(prediction_records, run_metadata)
    metrics = metric_payload(prediction_frame, n_bootstrap=n_bootstrap, seed=seed)
    write_csv(output_path / "test_predictions.csv", prediction_frame)
    write_json(output_path / "test_metrics.json", metrics)
    return prediction_frame, metrics

