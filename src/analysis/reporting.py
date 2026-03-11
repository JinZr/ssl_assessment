from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import write_text


def package_markdown_report(summary_dir: str | Path, figure_dir: str | Path, report_path: str | Path) -> str:
    summary_path = Path(summary_dir)
    figure_path = Path(figure_dir)
    best_encoder = pd.read_csv(summary_path / "best_per_encoder.csv")
    best_pair = pd.read_csv(summary_path / "best_per_pair.csv")
    significance = pd.read_csv(summary_path / "significance_tests.csv")
    lines = [
        "# SAP x QualiSpeech Experiment Report",
        "",
        "## Best Per Encoder",
        "",
        best_encoder.to_markdown(index=False),
        "",
        "## Best Per Pair",
        "",
        best_pair.to_markdown(index=False),
        "",
        "## Significance Tests",
        "",
        significance.to_markdown(index=False) if not significance.empty else "No significance results were produced.",
        "",
        "## Figures",
        "",
    ]
    for figure in sorted(figure_path.glob("*.png")):
        lines.append(f"### {figure.stem}")
        lines.append("")
        lines.append(f"![{figure.stem}]({figure})")
        lines.append("")
    write_text(report_path, "\n".join(lines))
    return str(report_path)

