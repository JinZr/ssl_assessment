from __future__ import annotations

import json

import pandas as pd

from src.data.parse_sap import parse_sap_dataset
from src.utils.io import read_parquet
from tests.conftest import write_wave


def _speaker_payload(speaker_id: str, filename: str, ratings: list[dict[str, str]] | None = None) -> dict[str, object]:
    return {
        "Contributor ID": speaker_id,
        "Etiology": "Parkinson's Disease",
        "BlockNumber": 6,
        "Files": [
            {
                "Filename": filename,
                "Created": "2023-04-12 12:21:32",
                "CreatedOrModified": "2023-04-13 19:10:28",
                "Comment": "",
                "Prompt": {
                    "Prompt Text": "Prompt",
                    "Transcript": "Prompt",
                    "Category Description": "Novel Sentences",
                    "Sub Category Description": "",
                },
                "Ratings": ratings or [],
            },
            {
                "Filename": f"{speaker_id}_empty.wav",
                "Created": "2023-04-12 12:22:00",
                "CreatedOrModified": "2023-04-13 19:15:00",
                "Comment": "",
                "Prompt": {
                    "Prompt Text": "Empty",
                    "Transcript": "Empty",
                    "Category Description": "Digital Assistant Commands",
                    "Sub Category Description": "",
                },
                "Ratings": [],
            },
        ],
    }


def test_parse_sap_dataset_handles_duplicates_and_canonicalization(tmp_path) -> None:
    train_dir = tmp_path / "train"
    dev_dir = tmp_path / "dev"
    train_speaker = train_dir / "speaker-a"
    dev_speaker = dev_dir / "speaker-b"
    train_speaker.mkdir(parents=True)
    dev_speaker.mkdir(parents=True)

    write_wave(train_speaker / "speaker-a_utt.wav")
    write_wave(train_speaker / "speaker-a_empty.wav")
    write_wave(dev_speaker / "speaker-b_utt.wav")
    write_wave(dev_speaker / "speaker-b_empty.wav")

    train_payload = _speaker_payload(
        "speaker-a",
        "speaker-a_utt.wav",
        ratings=[
            {"Level": "1", "Dimension Description": "Intelligbility "},
            {"Level": "4", "Dimension Description": "Naturalness"},
            {"Level": "2", "Dimension Description": "Breathy voice (continuous) "},
            {"Level": "", "Dimension Description": "Monopitch"},
        ],
    )
    dev_payload = _speaker_payload(
        "speaker-b",
        "speaker-b_utt.wav",
        ratings=[{"Level": "3", "Dimension Description": "Inappropriate silences"}],
    )

    (train_speaker / "speaker-a.json").write_text(json.dumps(train_payload), encoding="utf-8")
    (dev_speaker / "speaker-b.json").write_text(json.dumps(dev_payload), encoding="utf-8")
    (dev_dir / "speaker-b.json").write_text(json.dumps(dev_payload), encoding="utf-8")

    output_dir = tmp_path / "processed" / "sap"
    report = parse_sap_dataset(train_dir, dev_dir, output_dir)

    labels = read_parquet(output_dir / "sap_labels_long.parquet")
    utterances = read_parquet(output_dir / "sap_utterances.parquet")

    assert "intelligibility" in labels["dimension_canonical"].tolist()
    assert "breathy_voice_continuous" in labels["dimension_canonical"].tolist()
    assert report["splits"][0]["num_discarded_ratings"] == 1
    assert report["splits"][0]["discarded_rating_examples"][0]["raw_level"] == ""
    assert "monopitch" not in labels["dimension_canonical"].tolist()
    assert utterances["has_any_rating"].tolist().count(False) == 2
    assert report["splits"][1]["num_duplicates"] == 1
