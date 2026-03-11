from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.pipeline import run_experiment
from src.utils.config import load_yaml


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one resolved experiment config.")
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_yaml(Path(args.config))
    run_experiment(config)


if __name__ == "__main__":
    main()
