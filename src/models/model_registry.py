from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    paper_name: str
    model_name: str
    model_id: str
    family: str
    default_max_total_sec: int
    gradient_checkpointing: bool


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "w2v2_base": ModelSpec(
        paper_name="wav2vec 2.0 Base",
        model_name="w2v2_base",
        model_id="facebook/wav2vec2-base",
        family="wav2vec2",
        default_max_total_sec=180,
        gradient_checkpointing=False,
    ),
    "w2v2_large_lv60": ModelSpec(
        paper_name="wav2vec 2.0 Large*",
        model_name="w2v2_large_lv60",
        model_id="facebook/wav2vec2-large-lv60",
        family="wav2vec2",
        default_max_total_sec=90,
        gradient_checkpointing=True,
    ),
    "w2v2_large_robust": ModelSpec(
        paper_name="wav2vec 2.0 Large+",
        model_name="w2v2_large_robust",
        model_id="facebook/wav2vec2-large-robust",
        family="wav2vec2",
        default_max_total_sec=90,
        gradient_checkpointing=True,
    ),
    "hubert_base": ModelSpec(
        paper_name="HuBERT Base",
        model_name="hubert_base",
        model_id="facebook/hubert-base-ls960",
        family="hubert",
        default_max_total_sec=180,
        gradient_checkpointing=False,
    ),
    "hubert_large": ModelSpec(
        paper_name="HuBERT Large",
        model_name="hubert_large",
        model_id="facebook/hubert-large-ll60k",
        family="hubert",
        default_max_total_sec=90,
        gradient_checkpointing=True,
    ),
    "wavlm_base": ModelSpec(
        paper_name="WavLM Base",
        model_name="wavlm_base",
        model_id="microsoft/wavlm-base",
        family="wavlm",
        default_max_total_sec=180,
        gradient_checkpointing=False,
    ),
    "wavlm_base_plus": ModelSpec(
        paper_name="WavLM Base+",
        model_name="wavlm_base_plus",
        model_id="microsoft/wavlm-base-plus",
        family="wavlm",
        default_max_total_sec=180,
        gradient_checkpointing=False,
    ),
    "wavlm_large": ModelSpec(
        paper_name="WavLM Large",
        model_name="wavlm_large",
        model_id="microsoft/wavlm-large",
        family="wavlm",
        default_max_total_sec=90,
        gradient_checkpointing=True,
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model name: {model_name}")
    return MODEL_REGISTRY[model_name]
