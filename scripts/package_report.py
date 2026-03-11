from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.pipeline import _results_root
from src.analysis.reporting import package_markdown_report
from src.utils.config import load_yaml


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the final markdown report.")
    parser.add_argument("--config", default="configs/paths.yaml")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = REPO_ROOT
    config = load_yaml(repo_root / args.config)
    results_root = _results_root(config["paths"])
    package_markdown_report(results_root / "summaries", results_root / "figures", results_root / "reports" / "report.md")


if __name__ == "__main__":
    main()
