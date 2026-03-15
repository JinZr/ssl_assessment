from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.audio import load_audio


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index].to_dict()
        return row


@dataclass
class SpeechCollator:
    processor: Any
    sampling_rate: int = 16_000
    multitask_task_ids: list[str] | None = None
    max_input_sec: float | None = None

    def _segment_waveform(self, waveform: torch.Tensor) -> tuple[list[list[float]], list[float], list[float]]:
        if self.max_input_sec is None:
            duration_sec = waveform.numel() / self.sampling_rate
            return [waveform.detach().cpu().to(torch.float32).tolist()], [1.0], [duration_sec]
        max_input_samples = max(1, int(round(self.max_input_sec * self.sampling_rate)))
        if waveform.numel() <= max_input_samples:
            duration_sec = waveform.numel() / self.sampling_rate
            return [waveform.detach().cpu().to(torch.float32).tolist()], [1.0], [duration_sec]
        total_samples = int(waveform.numel())
        segments: list[list[float]] = []
        weights: list[float] = []
        durations_sec: list[float] = []
        start = 0
        while start < total_samples:
            end = min(start + max_input_samples, total_samples)
            segment = waveform[start:end]
            segments.append(segment.detach().cpu().to(torch.float32).tolist())
            weights.append(float(end - start) / float(total_samples))
            durations_sec.append((end - start) / self.sampling_rate)
            start = end
        return segments, weights, durations_sec

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        waveforms: list[list[float]] = []
        segment_parent_indices: list[int] = []
        segment_weights: list[float] = []
        segment_durations_sec: list[float] = []
        for item_index, item in enumerate(batch):
            waveform, _ = load_audio(item["audio_path"], target_sample_rate=self.sampling_rate)
            segments, weights, durations = self._segment_waveform(waveform)
            waveforms.extend(segments)
            segment_parent_indices.extend([item_index] * len(segments))
            segment_weights.extend(weights)
            segment_durations_sec.extend(durations)
        labels = torch.tensor(
            [float(item["label_for_loss"] if "label_for_loss" in item else item["label"]) for item in batch],
            dtype=torch.float32,
        )
        batch_dict = {
            "segment_waveforms": waveforms,
            "labels": labels,
            "metadata": batch,
            "segment_parent_indices": torch.tensor(segment_parent_indices, dtype=torch.long),
            "segment_weights": torch.tensor(segment_weights, dtype=torch.float32),
            "segment_durations_sec": torch.tensor(segment_durations_sec, dtype=torch.float32),
        }
        if "task_id" in batch[0]:
            batch_dict["task_ids"] = [item["task_id"] for item in batch]
        if "domain" in batch[0]:
            batch_dict["domains"] = [item["domain"] for item in batch]
        if self.multitask_task_ids:
            label_matrix = torch.zeros(len(batch), len(self.multitask_task_ids), dtype=torch.float32)
            label_mask = torch.zeros(len(batch), len(self.multitask_task_ids), dtype=torch.bool)
            for row_index, item in enumerate(batch):
                task_labels = item.get("task_labels") or {}
                for col_index, task_id in enumerate(self.multitask_task_ids):
                    value = task_labels.get(task_id)
                    if value is None or pd.isna(value):
                        continue
                    label_matrix[row_index, col_index] = float(value)
                    label_mask[row_index, col_index] = True
            batch_dict["task_labels"] = label_matrix
            batch_dict["task_mask"] = label_mask
            batch_dict["multitask_task_ids"] = list(self.multitask_task_ids)
            batch_dict["eval_task_ids"] = [item.get("eval_task_id") for item in batch]
        return batch_dict
