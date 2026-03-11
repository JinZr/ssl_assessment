from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import write_csv
from src.utils.metrics import compute_metrics


def add_severity_bin(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["severity_bin"] = pd.cut(
        enriched["y_true"],
        bins=[0, 1.5, 2.5, 3.5, 7.1],
        labels=["1", "2", "3", "4+"],
        include_lowest=True,
    )
    return enriched


def grouped_metric_table(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_value, group in frame.groupby(group_column, dropna=False, observed=False):
        if group.empty:
            continue
        metrics = compute_metrics(group["y_true"], group["y_pred"]).to_dict()
        clipped_metrics = compute_metrics(group["y_true"], group["y_pred_clipped"]).to_dict() if "y_pred_clipped" in group else {}
        rows.append(
            {
                "group": group_value,
                "count": len(group),
                **metrics,
                **{f"clipped_{key}": value for key, value in clipped_metrics.items()},
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def export_run_breakdowns(prediction_csv: str | Path, output_dir: str | Path) -> dict[str, str]:
    frame = pd.read_csv(prediction_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    if "prompt_category" in frame.columns and frame["prompt_category"].notna().any():
        prompt_table = grouped_metric_table(frame.dropna(subset=["prompt_category"]), "prompt_category")
        prompt_path = output_path / "prompt_breakdown.csv"
        write_csv(prompt_path, prompt_table)
        outputs["prompt"] = str(prompt_path)
    severity_table = grouped_metric_table(add_severity_bin(frame), "severity_bin")
    severity_path = output_path / "severity_breakdown.csv"
    write_csv(severity_path, severity_table)
    outputs["severity"] = str(severity_path)
    if "etiology" in frame.columns and frame["etiology"].notna().any():
        etiology_table = grouped_metric_table(frame.dropna(subset=["etiology"]), "etiology")
        etiology_path = output_path / "etiology_breakdown.csv"
        write_csv(etiology_path, etiology_table)
        outputs["etiology"] = str(etiology_path)
    return outputs
