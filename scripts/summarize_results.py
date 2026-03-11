from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.pipeline import _results_root, _runs_dir
from src.analysis.summarize import summarize_runs
from src.utils.config import load_yaml


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize all completed runs.")
    parser.add_argument("--config", default="configs/paths.yaml")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = REPO_ROOT
    config = load_yaml(repo_root / args.config)
    summarize_runs(_runs_dir(config["paths"]), _results_root(config["paths"]) / "summaries")


if __name__ == "__main__":
    main()
