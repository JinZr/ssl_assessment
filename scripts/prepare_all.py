from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.pipeline import _load_pair_configs, _load_task_configs, prepare_all
from src.utils.config import load_yaml


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare SAP and QualiSpeech manifests, splits, and pairs.")
    parser.add_argument("--config", default="configs/paths.yaml")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = REPO_ROOT
    paths_config = load_yaml(repo_root / args.config)
    task_configs = _load_task_configs(repo_root / "configs" / "tasks")
    pair_configs = _load_pair_configs(repo_root / "configs" / "pairs")
    prepare_all(paths_config, task_configs, pair_configs)


if __name__ == "__main__":
    main()
