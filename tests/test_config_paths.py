from __future__ import annotations

from pathlib import Path

from src.utils.config import load_config_bundle


def test_load_config_bundle_resolves_repo_relative_paths() -> None:
    repo_root = Path("/Users/zrjin/git/ssl_assessment")
    config = load_config_bundle(
        repo_root / "configs" / "defaults.yaml",
        repo_root / "configs" / "paths.yaml",
        extra_paths=[
            repo_root / "configs" / "models" / "wavlm_base.yaml",
            repo_root / "configs" / "experiments" / "baseline.yaml",
            repo_root / "configs" / "tasks" / "sap_naturalness.yaml",
        ],
    )

    assert config["paths"]["processed"]["splits_dir"] == str((repo_root / "data" / "processed" / "splits").resolve())
    assert config["model"]["cache_dir"] == str((repo_root / "data" / "cache" / "huggingface").resolve())

