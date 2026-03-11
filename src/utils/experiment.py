from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import write_json


def build_run_id(experiment: dict[str, Any]) -> str:
    fields = [
        experiment.get("split_protocol", "paper_faithful"),
        experiment.get("encoder"),
        experiment.get("method"),
        experiment.get("variant"),
        experiment.get("sap_target"),
        experiment.get("qs_aux", "none"),
        experiment.get("pair_id", "none"),
        f"ratio{experiment.get('ratio', 1.0)}",
        f"seed{experiment.get('seed')}",
    ]
    return "__".join(str(field).replace("/", "_") for field in fields if field is not None)


def run_complete(run_dir: str | Path) -> bool:
    target = Path(run_dir)
    return (target / "test_metrics.json").exists() and (target / "run_status.json").exists()


def write_run_status(run_dir: str | Path, status: str, extra: dict[str, Any] | None = None) -> None:
    payload = {"status": status}
    if extra:
        payload.update(extra)
    write_json(Path(run_dir) / "run_status.json", payload)
