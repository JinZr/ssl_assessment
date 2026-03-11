from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import write_csv, write_text


def _write_latex_table(path: Path, frame: pd.DataFrame) -> None:
    write_text(path, frame.to_latex(index=False, float_format="%.4f"))


def export_tables(summary_dir: str | Path, table_dir: str | Path) -> dict[str, str]:
    summary_path = Path(summary_dir)
    table_path = Path(table_dir)
    table_path.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(summary_path / "all_results_mean_std.csv")

    baseline = results[(results["method"] == "baseline") & (results["reviewer_control"].isna())].copy()
    table_a = baseline.pivot_table(index="sap_target", columns="encoder", values=["mse_mean", "lcc_mean", "srcc_mean"])
    table_a = table_a.reset_index()
    write_csv(table_path / "table_a_baseline.csv", table_a)
    _write_latex_table(table_path / "table_a_baseline.tex", table_a)

    main = results[
        (results["ratio"] == 1.0) & (results["method"].isin(["jt", "ft"])) & (results["reviewer_control"].isna())
    ].copy()
    table_b = main[
        [
            "pair_id",
            "method",
            "encoder",
            "mse_mean",
            "lcc_mean",
            "srcc_mean",
            "mse_std",
            "lcc_std",
            "srcc_std",
            "mse_gain_mean",
            "lcc_gain_mean",
            "srcc_gain_mean",
        ]
    ].sort_values(["pair_id", "method", "encoder"])
    write_csv(table_path / "table_b_main.csv", table_b)
    _write_latex_table(table_path / "table_b_main.tex", table_b)

    ratio_table = results[
        results["method"].isin(["jt", "ft"]) & results["reviewer_control"].isna()
    ].copy().sort_values(["pair_id", "encoder", "method", "ratio"])
    write_csv(table_path / "table_c_ratio_ablation.csv", ratio_table)
    _write_latex_table(table_path / "table_c_ratio_ablation.tex", ratio_table)
    ratio_detail_dir = table_path / "table_c_ratio_ablation_details"
    ratio_by_pair = ratio_table.groupby(["pair_id", "ratio"], dropna=False)[["mse_mean", "lcc_mean", "srcc_mean"]].mean().reset_index()
    ratio_by_encoder = ratio_table.groupby(["encoder", "ratio"], dropna=False)[["mse_mean", "lcc_mean", "srcc_mean"]].mean().reset_index()
    write_csv(table_path / "table_c_ratio_ablation_by_pair.csv", ratio_by_pair)
    write_csv(table_path / "table_c_ratio_ablation_by_encoder.csv", ratio_by_encoder)
    _write_latex_table(table_path / "table_c_ratio_ablation_by_pair.tex", ratio_by_pair)
    _write_latex_table(table_path / "table_c_ratio_ablation_by_encoder.tex", ratio_by_encoder)
    for (pair_id, encoder), detail in ratio_table.groupby(["pair_id", "encoder"], dropna=False):
        safe_pair = str(pair_id).replace("/", "_")
        safe_encoder = str(encoder).replace("/", "_")
        detail_path = ratio_detail_dir / f"{safe_pair}__{safe_encoder}.csv"
        write_csv(detail_path, detail)
        _write_latex_table(detail_path.with_suffix(".tex"), detail)

    reviewer = results[
        results["reviewer_control"].notna() | results["variant"].notna()
    ].copy()
    write_csv(table_path / "table_d_reviewer_controls.csv", reviewer)
    _write_latex_table(table_path / "table_d_reviewer_controls.tex", reviewer)

    return {
        "table_a": str(table_path / "table_a_baseline.csv"),
        "table_b": str(table_path / "table_b_main.csv"),
        "table_c": str(table_path / "table_c_ratio_ablation.csv"),
        "table_d": str(table_path / "table_d_reviewer_controls.csv"),
    }
