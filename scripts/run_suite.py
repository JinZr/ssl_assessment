from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.pipeline import _load_pair_configs, _load_task_configs, prepare_all, run_postprocessing, run_suite
from src.utils.cli import run_with_compact_errors
from src.utils.config import load_yaml


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an experiment suite YAML.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--full-traceback", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    def _main() -> None:
        repo_root = REPO_ROOT
        suite_config = load_yaml(repo_root / args.suite)
        paths_config = load_yaml(repo_root / args.config)
        task_configs = _load_task_configs(repo_root / "configs" / "tasks")
        pair_configs = _load_pair_configs(repo_root / "configs" / "pairs")
        if suite_config.get("prepare", False):
            prepare_all(paths_config, task_configs, pair_configs)
        run_suite(repo_root, repo_root / args.suite)
        if any(suite_config.get(flag, False) for flag in ("summarize", "tables", "figures", "report")):
            run_postprocessing(
                paths_config,
                run_summarize=suite_config.get("summarize", False),
                run_tables=suite_config.get("tables", False),
                run_figures=suite_config.get("figures", False),
                run_report=suite_config.get("report", False),
            )

    return run_with_compact_errors(_main, REPO_ROOT, full_traceback=args.full_traceback)


if __name__ == "__main__":
    raise SystemExit(main())
