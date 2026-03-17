from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.pipeline import run_experiment
from src.utils.cli import run_with_compact_errors
from src.utils.config import load_yaml


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one resolved experiment config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--full-traceback", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    def _main() -> None:
        config = load_yaml(Path(args.config))
        run_experiment(config)

    return run_with_compact_errors(_main, REPO_ROOT, full_traceback=args.full_traceback)


if __name__ == "__main__":
    raise SystemExit(main())
