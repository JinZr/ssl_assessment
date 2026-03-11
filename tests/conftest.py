from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile


def write_wave(path: str | Path, duration_sec: float = 0.25, sample_rate: int = 16_000) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    samples = int(duration_sec * sample_rate)
    time = np.linspace(0, duration_sec, samples, endpoint=False, dtype=np.float32)
    waveform = 0.1 * np.sin(2 * np.pi * 220 * time)
    waveform = (waveform * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(str(target), sample_rate, waveform)
    return target
