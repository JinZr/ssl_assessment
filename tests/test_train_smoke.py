from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import torch

from src.data.datasets import SpeechCollator
from src.models.regression import DualHeadSpeechRegressor
from src.models.hf_ssl_backbone import BackboneOutput
from src.samplers.dynamic_batch import DynamicDurationBatchSampler
from src.trainers.base import BaseTrainer
from src.trainers.baseline_trainer import BaselineTrainer
from tests.conftest import write_wave


class DummyProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, waveforms, sampling_rate: int, padding: bool, return_tensors: str):
        self.calls += 1
        waveforms = [torch.as_tensor(waveform, dtype=torch.float32) for waveform in waveforms]
        max_len = max(waveform.shape[0] for waveform in waveforms)
        batch = torch.zeros(len(waveforms), max_len)
        mask = torch.zeros(len(waveforms), max_len, dtype=torch.long)
        for index, waveform in enumerate(waveforms):
            batch[index, : waveform.shape[0]] = waveform
            mask[index, : waveform.shape[0]] = 1
        return {"input_values": batch, "attention_mask": mask}


class DummyHFBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.processor = DummyProcessor()
        self.hidden_size = 4
        self.model = SimpleNamespace()

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> BackboneOutput:
        hidden = input_values.unsqueeze(-1).repeat(1, 1, 4)
        if attention_mask is None:
            frame_mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
        else:
            frame_mask = attention_mask.bool()
        pooled = (hidden * frame_mask.unsqueeze(-1)).sum(1) / frame_mask.sum(1, keepdim=True)
        return BackboneOutput(hidden, None, frame_mask, pooled)


def test_baseline_trainer_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.trainers.base.BaseTrainer.build_backbone", lambda self: DummyHFBackbone())
    audio_dir = tmp_path / "audio"
    rows = []
    for index, label in enumerate([1.0, 2.0, 3.0, 4.0]):
        path = write_wave(audio_dir / f"utt{index}.wav")
        rows.append(
            {
                "utt_id": f"utt{index}",
                "speaker_id": f"spk{index}",
                "audio_path": str(path),
                "label": label,
                "label_for_loss": label,
                "label_min": 1.0,
                "label_max": 7.0,
                "duration_sec": 0.25,
                "target_dim": "naturalness",
                "domain": "sap",
            }
        )
    train_frame = pd.DataFrame(rows[:2])
    val_frame = pd.DataFrame(rows[2:3])
    test_frame = pd.DataFrame(rows[3:])
    config = {
        "experiment": {"seed": 13, "encoder": "wavlm_base", "method": "baseline", "sap_target": "naturalness"},
        "training": {"max_epochs": 1, "patience": 1, "gradient_accumulation_steps": 1, "max_total_sec": 180, "precision": "none", "lr": 1e-3},
        "model": {"name": "wavlm_base", "dropout": 0.1},
        "data": {"num_workers": 0, "sample_rate": 16_000},
        "evaluation": {"n_bootstrap": 10},
    }
    trainer = BaselineTrainer(config, tmp_path / "run")
    trainer.run(train_frame, val_frame, test_frame)
    trainer.cleanup()
    assert (tmp_path / "run" / "test_metrics.json").exists()
    assert (tmp_path / "run" / "best.ckpt").exists()


def test_speech_collator_falls_back_to_label_when_label_for_loss_missing(tmp_path) -> None:
    audio_path = write_wave(tmp_path / "audio" / "utt.wav")
    collator = SpeechCollator(processor=DummyProcessor(), sampling_rate=16_000)
    batch = [
        {
            "audio_path": str(audio_path),
            "label": 3.0,
        }
    ]
    output = collator(batch)
    assert torch.equal(output["labels"], torch.tensor([3.0], dtype=torch.float32))


def test_speech_collator_chunks_long_audio_by_max_input_sec(tmp_path) -> None:
    audio_path = write_wave(tmp_path / "audio" / "long.wav", duration_sec=0.25)
    collator = SpeechCollator(processor=DummyProcessor(), sampling_rate=16_000, max_input_sec=0.1)
    batch = [
        {
            "audio_path": str(audio_path),
            "label": 1.0,
        }
    ]
    output = collator(batch)
    assert len(output["segment_waveforms"]) == 3
    assert output["segment_parent_indices"].tolist() == [0, 0, 0]
    assert abs(float(output["segment_weights"].sum()) - 1.0) < 1e-6


def test_dynamic_batch_sampler_buckets_by_duration() -> None:
    sampler = DynamicDurationBatchSampler(
        durations=[9.0, 8.5, 1.0, 1.0, 1.0, 1.0],
        max_total_sec=10.0,
        shuffle=False,
        bucket_by_duration=True,
    )
    batches = list(iter(sampler))
    assert batches[0] == [0]
    assert batches[1] == [1, 2]


def test_trainer_encodes_long_audio_in_multiple_microbatches(tmp_path) -> None:
    audio_path = write_wave(tmp_path / "audio" / "long.wav", duration_sec=0.25)
    backbone = DummyHFBackbone()
    model = DualHeadSpeechRegressor(backbone=backbone)
    collator = SpeechCollator(processor=backbone.processor, sampling_rate=16_000, max_input_sec=0.1)
    batch = collator(
        [
            {
                "audio_path": str(audio_path),
                "label": 1.0,
                "domain": "sap",
            }
        ]
    )
    trainer = BaseTrainer(
        config={
            "experiment": {"seed": 13},
            "training": {"precision": "none", "max_input_sec": 0.1},
            "model": {"name": "wavlm_base", "max_input_sec": 0.1},
            "data": {"sample_rate": 16_000},
        },
        run_dir=tmp_path / "run",
    )
    with torch.no_grad():
        prediction = trainer._forward(model, batch, mode="dual", stage_cfg={"max_input_sec": 0.1})
    trainer.cleanup()
    assert prediction.shape == (1,)
    assert backbone.processor.calls == 3


def test_eval_and_finalize_use_stage_cfg_budget(tmp_path) -> None:
    audio_path = write_wave(tmp_path / "audio" / "long.wav", duration_sec=0.25)
    backbone = DummyHFBackbone()
    model = DualHeadSpeechRegressor(backbone=backbone)
    frame = pd.DataFrame(
        [
            {
                "utt_id": "utt0",
                "speaker_id": "spk0",
                "audio_path": str(audio_path),
                "label": 1.0,
                "label_min": 1.0,
                "label_max": 7.0,
                "duration_sec": 0.25,
                "target_dim": "naturalness",
                "domain": "sap",
            }
        ]
    )
    trainer = BaseTrainer(
        config={
            "experiment": {"seed": 13},
            "training": {"precision": "none", "max_total_sec": 0.25, "max_input_sec": 0.25, "gradient_accumulation_steps": 1},
            "model": {"name": "wavlm_base", "max_total_sec": 0.25, "max_input_sec": 0.25},
            "data": {"sample_rate": 16_000, "num_workers": 0},
            "evaluation": {"n_bootstrap": 4},
        },
        run_dir=tmp_path / "run",
    )
    stage_cfg = {"max_total_sec": 0.1, "max_input_sec": 0.1}

    _, _ = trainer.evaluate_frame(model, frame, mode="dual", stage_cfg=stage_cfg)
    assert backbone.processor.calls == 3

    backbone.processor.calls = 0
    trainer.finalize_run(model, test_frame=frame, mode="dual", run_metadata={"seed": 13}, stage_cfg=stage_cfg)
    trainer.cleanup()

    assert backbone.processor.calls == 3
    model_info = json.loads((tmp_path / "run" / "model_info.json").read_text(encoding="utf-8"))
    assert model_info["max_total_sec"] == 0.1
    assert model_info["max_input_sec"] == 0.1
