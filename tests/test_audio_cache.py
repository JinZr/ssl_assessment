from __future__ import annotations

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

