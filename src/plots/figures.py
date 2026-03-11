from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ENCODER_SCALE = {
    "wavlm_base": 1,
    "wavlm_base_plus": 1.2,
    "wavlm_large": 3,
    "w2v2_base": 1,
    "w2v2_large_lv60": 3,
    "w2v2_large_robust": 3,
    "hubert_base": 1,
    "hubert_large": 3,
}


def _save(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def _load_best_run_breakdown(
    summary_path: Path,
    *,
    by: str,
    value: str,
) -> tuple[pd.Series, pd.Series | None]:
    results = pd.read_csv(summary_path / "all_results_long.csv")
    breakdown_index = pd.read_csv(summary_path / "breakdown_index.csv").fillna("")
    subset = results[
        (results[by] == value)
        & (results["ratio"] == 1.0)
        & (results["method"].isin(["jt", "ft"]))
        & (results["reviewer_control"].isna())
    ].sort_values(["mse", "lcc", "srcc"], ascending=[True, False, False])
    if subset.empty:
        raise ValueError(f"No non-baseline results found for {by}={value}")
    best_run = subset.iloc[0]
    baseline = results[
        (results["method"] == "baseline")
        & (results["variant"].isna())
        & (results["sap_target"] == best_run["sap_target"])
        & (results["encoder"] == best_run["encoder"])
        & (results["seed"] == best_run["seed"])
        & (results["split_protocol"] == best_run["split_protocol"])
    ]
    baseline_run = baseline.iloc[0] if not baseline.empty else None
    best_breakdown = breakdown_index[breakdown_index["run_id"] == best_run["run_id"]].iloc[0]
    baseline_breakdown = None
    if baseline_run is not None:
        baseline_rows = breakdown_index[breakdown_index["run_id"] == baseline_run["run_id"]]
        if not baseline_rows.empty:
            baseline_breakdown = baseline_rows.iloc[0]
    return best_breakdown, baseline_breakdown


def _export_breakdown_comparison(
    best_path: str,
    baseline_path: str | None,
    output_stem: Path,
    title: str,
    x_label: str,
) -> None:
    best_frame = pd.read_csv(best_path)
    merged = best_frame.rename(columns={"group": x_label, "mse": "best_mse"})[[x_label, "best_mse"]]
    if baseline_path:
        baseline_frame = pd.read_csv(baseline_path)
        merged = merged.merge(
            baseline_frame.rename(columns={"group": x_label, "mse": "baseline_mse"})[[x_label, "baseline_mse"]],
            on=x_label,
            how="left",
        )
    fig, ax = plt.subplots(figsize=(8, 4))
    if "baseline_mse" in merged.columns:
        ax.bar(merged[x_label], merged["baseline_mse"], label="baseline", alpha=0.7)
    ax.bar(merged[x_label], merged["best_mse"], label="best_transfer", alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("MSE")
    ax.set_xlabel(x_label.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    _save(fig, output_stem)


def export_ratio_figures(summary_dir: str | Path, figure_dir: str | Path) -> None:
    summary_path = Path(summary_dir)
    figure_path = Path(figure_dir)
    ratio_curves = pd.read_csv(summary_path / "ratio_curves.csv")
    for metric in ("mse_mean", "lcc_mean", "srcc_mean"):
        for pair_id, pair_frame in ratio_curves.groupby("pair_id", dropna=True):
            fig, ax = plt.subplots(figsize=(7, 4))
            for (method, encoder), group in pair_frame.groupby(["method", "encoder"]):
                group = group.sort_values("ratio")
                ax.plot(group["ratio"], group[metric], marker="o", label=f"{method}:{encoder}")
            ax.set_title(f"{pair_id} {metric.replace('_mean', '').upper()}")
            ax.set_xlabel("Aux Ratio")
            ax.set_ylabel(metric.replace("_mean", "").upper())
            ax.legend(fontsize=7, ncols=2)
            _save(fig, figure_path / f"fig_ratio_ablation_{metric.replace('_mean', '')}_{pair_id}")


def export_gain_figures(summary_dir: str | Path, figure_dir: str | Path) -> None:
    summary_path = Path(summary_dir)
    figure_path = Path(figure_dir)
    results = pd.read_csv(summary_path / "all_results_mean_std.csv")
    baseline = results[(results["method"] == "baseline") & (results["reviewer_control"].isna())][["encoder", "sap_target", "mse_mean"]].rename(
        columns={"mse_mean": "baseline_mse_mean"}
    )
    main = results[results["method"].isin(["jt", "ft"]) & results["reviewer_control"].isna()]
    merged = main.merge(baseline, on=["encoder", "sap_target"], how="left")
    merged["mse_gain"] = merged["baseline_mse_mean"] - merged["mse_mean"]
    for sap_target, group in merged.groupby("sap_target"):
        fig, ax = plt.subplots(figsize=(8, 4))
        for method, method_frame in group.groupby("method"):
            ax.bar(
                [f"{method}:{encoder}" for encoder in method_frame["encoder"]],
                method_frame["mse_gain"],
                label=method,
            )
        ax.set_title(f"Gain over baseline: {sap_target}")
        ax.set_ylabel("Baseline MSE - Method MSE")
        ax.tick_params(axis="x", rotation=75)
        _save(fig, figure_path / f"fig_gain_over_baseline_{sap_target}")

    scale_frame = merged.groupby("encoder", as_index=False)["mse_gain"].mean()
    scale_frame["encoder_scale"] = scale_frame["encoder"].map(ENCODER_SCALE)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(scale_frame["encoder_scale"], scale_frame["mse_gain"])
    for row in scale_frame.itertuples():
        ax.annotate(row.encoder, (row.encoder_scale, row.mse_gain))
    ax.set_xlabel("Relative encoder scale")
    ax.set_ylabel("Mean MSE gain")
    ax.set_title("Encoder scale vs gain")
    _save(fig, figure_path / "fig_encoder_scale_vs_gain")


def export_prediction_figures(results_runs_dir: str | Path, figure_dir: str | Path) -> None:
    figure_path = Path(figure_dir)
    for run_dir in sorted(Path(results_runs_dir).iterdir()):
        prediction_path = run_dir / "test_predictions.csv"
        if not prediction_path.exists():
            continue
        frame = pd.read_csv(prediction_path)
        if frame.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(frame["y_true"], frame["y_pred"], alpha=0.5)
        ax.set_title(run_dir.name)
        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Prediction")
        _save(fig, figure_path / f"scatter_{run_dir.name}")

        fig, ax = plt.subplots(figsize=(6, 4))
        residuals = frame["y_pred"] - frame["y_true"]
        ax.hist(residuals, bins=20)
        ax.set_title(f"Residual histogram: {run_dir.name}")
        ax.set_xlabel("Residual")
        _save(fig, figure_path / f"residual_hist_{run_dir.name}")


def export_breakdown_figures(summary_dir: str | Path, figure_dir: str | Path) -> None:
    summary_path = Path(summary_dir)
    figure_path = Path(figure_dir)
    best_pairs = pd.read_csv(summary_path / "best_per_pair.csv")
    best_targets = pd.read_csv(summary_path / "best_per_target.csv")

    for pair_id in best_pairs["pair_id"].dropna().unique():
        best_breakdown, baseline_breakdown = _load_best_run_breakdown(summary_path, by="pair_id", value=pair_id)
        if best_breakdown.get("prompt_path"):
            _export_breakdown_comparison(
                best_breakdown["prompt_path"],
                baseline_breakdown["prompt_path"] if baseline_breakdown is not None and baseline_breakdown.get("prompt_path") else None,
                figure_path / f"fig_prompt_breakdown_{pair_id}",
                title=f"Prompt breakdown: {pair_id}",
                x_label="prompt_category",
            )
        if best_breakdown.get("etiology_path"):
            _export_breakdown_comparison(
                best_breakdown["etiology_path"],
                baseline_breakdown["etiology_path"] if baseline_breakdown is not None and baseline_breakdown.get("etiology_path") else None,
                figure_path / f"fig_etiology_breakdown_{pair_id}",
                title=f"Etiology breakdown: {pair_id}",
                x_label="etiology",
            )

    for sap_target in best_targets["sap_target"].dropna().unique():
        best_breakdown, baseline_breakdown = _load_best_run_breakdown(summary_path, by="sap_target", value=sap_target)
        if best_breakdown.get("severity_path"):
            _export_breakdown_comparison(
                best_breakdown["severity_path"],
                baseline_breakdown["severity_path"] if baseline_breakdown is not None and baseline_breakdown.get("severity_path") else None,
                figure_path / f"fig_severity_bin_{sap_target}",
                title=f"Severity-bin breakdown: {sap_target}",
                x_label="severity_bin",
            )
