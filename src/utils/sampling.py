from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def exact_proportional_allocation(
    counts: dict[Any, int],
    target_n: int,
    *,
    replace: bool = False,
) -> dict[Any, int]:
    if target_n < 0:
        raise ValueError("target_n must be non-negative")
    total = sum(counts.values())
    if total == 0:
        return {key: 0 for key in counts}
    if not replace and target_n > total:
        raise ValueError("Cannot sample more items than available without replacement.")

    ideal = {key: (value / total) * target_n for key, value in counts.items()}
    allocations = {
        key: int(np.floor(ideal[key])) if replace else min(counts[key], int(np.floor(ideal[key])))
        for key in counts
    }
    remaining = target_n - sum(allocations.values())
    if remaining <= 0:
        return allocations

    while remaining > 0:
        candidates = [key for key in counts if replace or allocations[key] < counts[key]]
        if not candidates:
            break
        candidates.sort(
            key=lambda key: (
                -(ideal[key] - allocations[key]),
                -(counts[key] - allocations[key] if not replace else counts[key]),
                str(key),
            )
        )
        allocations[candidates[0]] += 1
        remaining -= 1
    return allocations


def sample_stratified_exact(
    frame: pd.DataFrame,
    label_column: str,
    sample_n: int,
    seed: int,
    *,
    replace: bool = False,
) -> pd.DataFrame:
    if sample_n == 0:
        return frame.iloc[0:0].copy()
    if sample_n < 0:
        raise ValueError("sample_n must be non-negative")
    if frame.empty:
        raise ValueError("Cannot sample from an empty frame.")
    if not replace and sample_n > len(frame):
        raise ValueError("Requested sample_n exceeds frame size for sampling without replacement.")

    rng = np.random.default_rng(seed)
    counts = frame[label_column].value_counts(dropna=False).to_dict()
    allocations = exact_proportional_allocation(counts, sample_n, replace=replace)
    sampled_parts: list[pd.DataFrame] = []
    grouped = frame.groupby(label_column, dropna=False, sort=True)
    for label, group in grouped:
        take_n = allocations.get(label, 0)
        if take_n <= 0:
            continue
        sampled_index = rng.choice(group.index.to_numpy(), size=take_n, replace=replace)
        sampled_parts.append(group.loc[sampled_index])
    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else frame.iloc[0:0].copy()
    if len(sampled) != sample_n:
        raise ValueError(f"Sampled {len(sampled)} rows but expected {sample_n}.")
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@dataclass(frozen=True)
class SpeakerSubsetResult:
    speaker_ids: list[str]
    utterance_count: int
    label_l1_distance: float


def select_speaker_subset(
    frame: pd.DataFrame,
    speaker_column: str,
    label_column: str,
    target_utterances: int,
    seed: int,
    num_restarts: int = 64,
) -> SpeakerSubsetResult:
    if frame.empty:
        raise ValueError("Cannot select speakers from an empty frame.")
    if target_utterances <= 0:
        raise ValueError("target_utterances must be positive.")

    label_order = sorted(frame[label_column].dropna().unique().tolist())
    global_distribution = (
        frame[label_column].value_counts(normalize=True, dropna=False).reindex(label_order, fill_value=0.0).to_numpy(dtype=float)
    )

    speaker_stats = {}
    for speaker_id, group in frame.groupby(speaker_column, dropna=False):
        speaker_stats[str(speaker_id)] = {
            "count": len(group),
            "dist": group[label_column].value_counts(normalize=True, dropna=False).reindex(label_order, fill_value=0.0).to_numpy(dtype=float),
        }

    speaker_ids = sorted(speaker_stats)
    rng = np.random.default_rng(seed)
    best_key: tuple[float, float, float] | None = None
    best_result: SpeakerSubsetResult | None = None

    def score(selected: list[str]) -> tuple[float, int, float]:
        count = sum(speaker_stats[speaker]["count"] for speaker in selected)
        distribution = np.zeros(len(label_order), dtype=float)
        total_count = 0
        for speaker in selected:
            speaker_count = speaker_stats[speaker]["count"]
            distribution += speaker_stats[speaker]["dist"] * speaker_count
            total_count += speaker_count
        if total_count > 0:
            distribution /= total_count
        label_l1 = float(np.abs(distribution - global_distribution).sum())
        count_penalty = abs(count - target_utterances) / max(1, target_utterances)
        return count_penalty + label_l1, count, label_l1

    for restart in range(max(1, num_restarts)):
        candidates = speaker_ids.copy()
        rng.shuffle(candidates)
        selected: list[str] = []
        remaining = set(candidates)
        current_score, current_count, current_l1 = score(selected)
        while remaining and current_count < target_utterances:
            best_candidate: tuple[float, str, int, float] | None = None
            for speaker in list(remaining):
                trial_selected = selected + [speaker]
                trial_score, trial_count, trial_l1 = score(trial_selected)
                candidate_tuple = (trial_score, speaker, trial_count, trial_l1)
                if best_candidate is None or candidate_tuple < best_candidate:
                    best_candidate = candidate_tuple
            assert best_candidate is not None
            _, speaker, current_count, current_l1 = best_candidate
            selected.append(speaker)
            remaining.remove(speaker)
            current_score = best_candidate[0]

        improved = True
        while improved and len(selected) > 1:
            improved = False
            for speaker in list(selected):
                trial_selected = [item for item in selected if item != speaker]
                if not trial_selected:
                    continue
                trial_score, trial_count, trial_l1 = score(trial_selected)
                if trial_score < current_score:
                    selected = trial_selected
                    current_score, current_count, current_l1 = trial_score, trial_count, trial_l1
                    improved = True
                    break

        result = SpeakerSubsetResult(
            speaker_ids=sorted(selected),
            utterance_count=current_count,
            label_l1_distance=current_l1,
        )
        result_key = (current_score, abs(current_count - target_utterances), current_l1)
        if best_key is None or result_key < best_key:
            best_key = result_key
            best_result = result

    assert best_result is not None
    return best_result
