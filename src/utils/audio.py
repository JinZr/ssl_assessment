from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from tqdm.auto import tqdm

try:
    import torchaudio
except Exception:
    torchaudio = None


TARGET_SAMPLE_RATE = 16_000
DEFAULT_AUDIO_PROBE_WORKERS = max(1, min(8, os.cpu_count() or 1))


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


def _probe_audio_uncached(path: str | Path) -> dict[str, float | int | None]:
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


def probe_audio(path: str | Path) -> dict[str, float | int | None]:
    return _probe_audio_uncached(path)


def _audio_probe_signature(path: str | Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}


def _load_audio_probe_cache(cache_path: str | Path | None) -> dict[str, Any]:
    if cache_path is None:
        return {"version": 1, "entries": {}}
    target = Path(cache_path)
    if not target.exists():
        return {"version": 1, "entries": {}}
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_audio_probe_cache(cache_path: str | Path | None, payload: dict[str, Any]) -> None:
    if cache_path is None:
        return
    target = Path(cache_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _probe_audio_worker(path_str: str) -> tuple[str, dict[str, float | int | None]]:
    return path_str, _probe_audio_uncached(path_str)


def probe_audio_many(
    paths: list[str | Path],
    *,
    cache_path: str | Path | None = None,
    max_workers: int | None = None,
    desc: str | None = None,
) -> dict[str, dict[str, float | int | None]]:
    unique_paths = [str(Path(path)) for path in dict.fromkeys(paths)]
    results: dict[str, dict[str, float | int | None]] = {}
    cache_payload = _load_audio_probe_cache(cache_path)
    cache_entries: dict[str, Any] = cache_payload.setdefault("entries", {})
    cache_updated = False
    misses: list[str] = []

    for path_str in unique_paths:
        target = Path(path_str)
        if not target.exists():
            results[path_str] = {"duration_sec": None, "sample_rate": None, "num_samples": None}
            continue
        signature = _audio_probe_signature(target)
        cached = cache_entries.get(path_str)
        if cached and cached.get("signature") == signature:
            results[path_str] = cached["stats"]
            continue
        misses.append(path_str)

    worker_count = max_workers if max_workers is not None else DEFAULT_AUDIO_PROBE_WORKERS
    worker_count = max(1, worker_count)
    if misses:
        if worker_count == 1 or len(misses) == 1:
            iterator = tqdm(misses, desc=desc or "Probe audio", unit="file")
            for path_str in iterator:
                stats = _probe_audio_uncached(path_str)
                results[path_str] = stats
                cache_entries[path_str] = {"signature": _audio_probe_signature(path_str), "stats": stats}
                cache_updated = True
        else:
            def run_parallel(executor_cls):
                with executor_cls(max_workers=worker_count) as executor:
                    futures = {executor.submit(_probe_audio_worker, path_str): path_str for path_str in misses}
                    progress = tqdm(total=len(futures), desc=desc or "Probe audio", unit="file")
                    try:
                        for future in as_completed(futures):
                            path_str, stats = future.result()
                            results[path_str] = stats
                            cache_entries[path_str] = {"signature": _audio_probe_signature(path_str), "stats": stats}
                            progress.update(1)
                        return True
                    finally:
                        progress.close()

            try:
                cache_updated = run_parallel(ProcessPoolExecutor) or cache_updated
            except (OSError, PermissionError, NotImplementedError):
                cache_updated = run_parallel(ThreadPoolExecutor) or cache_updated

    if cache_updated:
        _save_audio_probe_cache(cache_path, cache_payload)
    return results
