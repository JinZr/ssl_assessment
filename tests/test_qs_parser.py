from __future__ import annotations

import pandas as pd

from src.data.parse_qualispeech import parse_qualispeech_dataset
from src.utils.io import read_parquet
from tests.conftest import write_wave


def test_parse_qualispeech_dataset_normalizes_columns(tmp_path) -> None:
    root = tmp_path / "qualispeech"
    for split in ("train", "val", "test"):
        split_dir = root / split
        split_dir.mkdir(parents=True)
        write_wave(split_dir / f"{split}_sample.wav")
        frame = pd.DataFrame(
            [
                {
                    "id": f"{split}_sample.wav",
                    "Naturalness": 5,
                    "Background noise": 4,
                    "Listening effort": 3,
                    "Overall quality": 4,
                    "Unnatural pause": "Very smooth",
                }
            ]
        )
        frame.to_csv(root / f"{split}.csv", index=False)

    parse_qualispeech_dataset(root, root / "processed")
    train = read_parquet(root / "processed" / "qs_train.parquet")
    assert "background_noise" in train.columns
    assert "listening_effort" in train.columns
    assert train.loc[0, "utt_id"] == "train_sample"
