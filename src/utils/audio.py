from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

try:
    import torchaudio
except Exception:
    torchaudio = None


TARGET_SAMPLE_RATE = 16_000


def load_audio(path: str | Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[torch.Tensor, int]:
    if torchaudio is not None:
        waveform, sample_rate = torchaudio.load(str(path))
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
        waveform = waveform.to(torch.float32)
        return waveform.squeeze(0), sample_rate

    sample_rate, waveform_np = wavfile.read(str(path))
    original_dtype = waveform_np.dtype
    waveform_np = waveform_np.astype(np.float32)
    if waveform_np.ndim == 2:
        waveform_np = waveform_np.mean(axis=1)
    if original_dtype.kind in {"i", "u"}:
        waveform_np /= np.iinfo(original_dtype).max
    if sample_rate != target_sample_rate:
        waveform_np = resample_poly(waveform_np, target_sample_rate, sample_rate).astype(np.float32)
        sample_rate = target_sample_rate
    return torch.tensor(waveform_np, dtype=torch.float32), sample_rate


def probe_audio(path: str | Path) -> dict[str, float | int | None]:
    target = Path(path)
    if not target.exists():
        return {"duration_sec": None, "sample_rate": None, "num_samples": None}
    if torchaudio is not None:
        try:
            if hasattr(torchaudio, "info"):
                info = torchaudio.info(str(target))
                return {
                    "duration_sec": info.num_frames / info.sample_rate if info.sample_rate else None,
                    "sample_rate": info.sample_rate,
                    "num_samples": info.num_frames,
                }
            if hasattr(torchaudio, "load"):
                waveform, sample_rate = torchaudio.load(str(target))
                num_samples = int(waveform.shape[-1])
                return {
                    "duration_sec": num_samples / sample_rate if sample_rate else None,
                    "sample_rate": int(sample_rate),
                    "num_samples": num_samples,
                }
        except Exception:
            pass
    sample_rate, waveform_np = wavfile.read(str(target))
    num_samples = int(waveform_np.shape[0])
    return {
        "duration_sec": num_samples / sample_rate if sample_rate else None,
        "sample_rate": int(sample_rate),
        "num_samples": num_samples,
    }
