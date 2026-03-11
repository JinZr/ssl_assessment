from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.pipeline import run_suite


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an experiment suite YAML.")
    parser.add_argument("--suite", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = REPO_ROOT
    run_suite(repo_root, repo_root / args.suite)


if __name__ == "__main__":
    main()
