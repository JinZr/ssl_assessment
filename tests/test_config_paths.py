from __future__ import annotations

from pathlib import Path

from src.utils.config import load_config_bundle
from src.utils.hf import resolved_revision_record


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


def test_load_config_bundle_keeps_large_model_input_cap() -> None:
    repo_root = Path("/Users/zrjin/git/ssl_assessment")
    config = load_config_bundle(
        repo_root / "configs" / "defaults.yaml",
        repo_root / "configs" / "paths.yaml",
        extra_paths=[
            repo_root / "configs" / "models" / "wavlm_large.yaml",
            repo_root / "configs" / "experiments" / "baseline.yaml",
            repo_root / "configs" / "tasks" / "sap_naturalness.yaml",
        ],
    )

    assert config["model"]["max_input_sec"] == 90


def test_resolved_revision_record_skips_hub_lookup_when_local_files_only() -> None:
    record = resolved_revision_record(
        "wavlm_base",
        "microsoft/wavlm-base",
        "main",
        local_files_only=True,
    )

    assert record["requested_revision"] == "main"
    assert record["resolved_revision"] == "main"
