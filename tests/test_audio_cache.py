from __future__ import annotations

import torch

from src.utils import audio
from tests.conftest import write_wave


def test_probe_audio_many_reuses_cache(tmp_path, monkeypatch) -> None:
    audio_path = write_wave(tmp_path / "sample.wav")
    cache_path = tmp_path / "audio_probe_cache.json"
    call_count = {"count": 0}

    def fake_probe(path):  # noqa: ANN001
        call_count["count"] += 1
        return {"duration_sec": 0.25, "sample_rate": 16_000, "num_samples": 4_000}

    monkeypatch.setattr(audio, "_probe_audio_uncached", fake_probe)

    first = audio.probe_audio_many([audio_path], cache_path=cache_path, max_workers=1)
    second = audio.probe_audio_many([audio_path], cache_path=cache_path, max_workers=1)

    assert call_count["count"] == 1
    assert str(audio_path) in first
    assert first == second


def test_load_audio_falls_back_when_torchaudio_backend_is_unavailable(tmp_path, monkeypatch) -> None:
    audio_path = write_wave(tmp_path / "sample.wav", sample_rate=8_000)

    class BrokenTorchaudio:
        @staticmethod
        def load(path):  # noqa: ANN001
            raise RuntimeError("Couldn't find appropriate backend")

    monkeypatch.setattr(audio, "torchaudio", BrokenTorchaudio())

    waveform, sample_rate = audio.load_audio(audio_path, target_sample_rate=16_000)

    assert sample_rate == 16_000
    assert waveform.dtype == torch.float32
    assert waveform.ndim == 1
    assert waveform.numel() == 4_000


def test_load_audio_skips_torchaudio_for_wav_inputs(tmp_path, monkeypatch) -> None:
    audio_path = write_wave(tmp_path / "sample.wav", sample_rate=8_000)
    attempts = {"load": 0}

    class FakeTorchaudio:
        @staticmethod
        def load(path):  # noqa: ANN001
            attempts["load"] += 1
            raise AssertionError("WAV inputs should not go through torchaudio.load")

    monkeypatch.setattr(audio, "torchaudio", FakeTorchaudio())

    waveform, sample_rate = audio.load_audio(audio_path, target_sample_rate=16_000)

    assert attempts["load"] == 0
    assert sample_rate == 16_000
    assert waveform.dtype == torch.float32
    assert waveform.ndim == 1
    assert waveform.numel() == 4_000
