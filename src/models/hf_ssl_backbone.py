from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoProcessor,
    Wav2Vec2FeatureExtractor,
)

from src.models.model_registry import get_model_spec
from src.utils.hf import retry


@dataclass
class BackboneOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None
    frame_mask: torch.Tensor
    pooled_embedding: torch.Tensor


class HFSSLBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        revision: str | None = None,
        apply_spec_augment: bool = False,
        layerdrop: float = 0.0,
        output_hidden_states: bool = False,
        gradient_checkpointing: bool | None = None,
    ) -> None:
        super().__init__()
        self.spec = get_model_spec(model_name)
        self.model_name = model_name
        self.model_id = self.spec.model_id
        self.revision = revision
        self.output_hidden_states = output_hidden_states

        config = retry(
            lambda: AutoConfig.from_pretrained(self.model_id, cache_dir=cache_dir, revision=revision)
        )
        if hasattr(config, "apply_spec_augment"):
            config.apply_spec_augment = apply_spec_augment
        if hasattr(config, "layerdrop"):
            config.layerdrop = layerdrop

        self.model = retry(
            lambda: AutoModel.from_pretrained(
                self.model_id,
                cache_dir=cache_dir,
                revision=revision,
                config=config,
            )
        )
        if gradient_checkpointing or (gradient_checkpointing is None and self.spec.gradient_checkpointing):
            self.model.gradient_checkpointing_enable()
        self.processor = self._load_processor(cache_dir=cache_dir, revision=revision)
        self.hidden_size = int(config.hidden_size)
        self.config_dict = config.to_dict()

    def _load_processor(self, cache_dir: str | None, revision: str | None) -> Any:
        try:
            return retry(lambda: AutoProcessor.from_pretrained(self.model_id, cache_dir=cache_dir, revision=revision))
        except Exception:
            try:
                return retry(
                    lambda: AutoFeatureExtractor.from_pretrained(
                        self.model_id,
                        cache_dir=cache_dir,
                        revision=revision,
                    )
                )
            except Exception:
                return retry(
                    lambda: Wav2Vec2FeatureExtractor.from_pretrained(
                        self.model_id,
                        cache_dir=cache_dir,
                        revision=revision,
                    )
                )

    def _feature_lengths_from_attention_mask(
        self,
        output_length: int,
        attention_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if hasattr(self.model, "_get_feature_vector_attention_mask"):
            return self.model._get_feature_vector_attention_mask(output_length, attention_mask).to(device=device)

        if hasattr(self.model, "_get_feat_extract_output_lengths"):
            lengths = self.model._get_feat_extract_output_lengths(attention_mask.sum(-1))
        else:
            lengths = attention_mask.sum(-1)
            kernels = getattr(self.model.config, "conv_kernel", ())
            strides = getattr(self.model.config, "conv_stride", ())
            for kernel, stride in zip(kernels, strides):
                lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
            lengths = torch.clamp(lengths, min=0)

        positions = torch.arange(output_length, device=device).unsqueeze(0)
        return positions < lengths.unsqueeze(1)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> BackboneOutput:
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=self.output_hidden_states,
        )
        hidden_states = outputs.last_hidden_state
        if attention_mask is None:
            frame_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        else:
            frame_mask = self._feature_lengths_from_attention_mask(
                hidden_states.shape[1],
                attention_mask,
                hidden_states.device,
            )
        pooled_embedding = self.masked_mean_pool(hidden_states, frame_mask)
        return BackboneOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states if self.output_hidden_states else None,
            frame_mask=frame_mask,
            pooled_embedding=pooled_embedding,
        )

    @staticmethod
    def masked_mean_pool(hidden_states: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        mask = frame_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom
