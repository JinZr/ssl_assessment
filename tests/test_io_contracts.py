from __future__ import annotations

import pandas as pd

from src.utils.io import read_json, read_parquet, write_parquet


def test_write_parquet_uses_explicit_fallback_metadata_on_failure(tmp_path, monkeypatch) -> None:
    frame = pd.DataFrame({"a": [1, 2]})

    def fail(*args, **kwargs):  # noqa: ANN001, ARG001
        raise RuntimeError("parquet unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    target = tmp_path / "table.parquet"
    write_parquet(target, frame)

    meta = read_json(target.with_suffix(".parquet.meta.json"))
    assert meta["actual_format"] == "pickle"
    assert target.with_suffix(".parquet.pkl").exists()
    restored = read_parquet(target)
    assert restored.equals(frame)

