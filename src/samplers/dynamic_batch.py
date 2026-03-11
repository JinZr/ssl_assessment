from __future__ import annotations

import math
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class DynamicDurationBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        durations: list[float],
        max_total_sec: float,
        shuffle: bool = True,
        seed: int = 13,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.durations = [duration or 0.0 for duration in durations]
        self.max_total_sec = max_total_sec
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        indices = np.arange(len(self.durations))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(indices)
        if self.world_size > 1:
            indices = indices[self.rank :: self.world_size]
        batch: list[int] = []
        total_sec = 0.0
        for index in indices.tolist():
            duration = max(self.durations[index], 0.01)
            if batch and total_sec + duration > self.max_total_sec:
                yield batch
                batch = []
                total_sec = 0.0
            batch.append(index)
            total_sec += duration
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total = sum(duration or 0.0 for duration in self.durations)
        if total == 0:
            return len(self.durations)
        shard_total = total / self.world_size
        return max(1, math.ceil(shard_total / self.max_total_sec))

