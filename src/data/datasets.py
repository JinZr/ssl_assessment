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

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        waveforms: list[torch.Tensor] = []
        for item in batch:
            waveform, _ = load_audio(item["audio_path"], target_sample_rate=self.sampling_rate)
            waveforms.append(waveform)
        features = self.processor(
            waveforms,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        labels = torch.tensor([float(item["label_for_loss"]) for item in batch], dtype=torch.float32)
        batch_dict = {
            "input_values": features["input_values"],
            "attention_mask": features.get("attention_mask"),
            "labels": labels,
            "metadata": batch,
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
