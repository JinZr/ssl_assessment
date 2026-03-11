from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from src.models.hf_ssl_backbone import HFSSLBackbone


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings).squeeze(-1)


class SpeechRegressor(nn.Module):
    def __init__(self, backbone: HFSSLBackbone, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = RegressionHead(
            input_dim=backbone.hidden_size,
            hidden_dim=hidden_dim or backbone.hidden_size,
            dropout=dropout,
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        features = self.backbone(input_values=input_values, attention_mask=attention_mask)
        prediction = self.head(features.pooled_embedding)
        return {
            "prediction": prediction,
            "pooled_embedding": features.pooled_embedding,
            "frame_mask": features.frame_mask,
            "last_hidden_state": features.last_hidden_state,
        }


class DualHeadSpeechRegressor(nn.Module):
    def __init__(self, backbone: HFSSLBackbone, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        shared_hidden_dim = hidden_dim or backbone.hidden_size
        self.backbone = backbone
        self.sap_head = RegressionHead(backbone.hidden_size, shared_hidden_dim, dropout=dropout)
        self.qs_head = RegressionHead(backbone.hidden_size, shared_hidden_dim, dropout=dropout)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        domains: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(input_values=input_values, attention_mask=attention_mask)
        sap_pred = self.sap_head(features.pooled_embedding)
        qs_pred = self.qs_head(features.pooled_embedding)
        if domains is None:
            prediction = sap_pred
        else:
            prediction = torch.where(
                torch.tensor([domain == "sap" for domain in domains], device=sap_pred.device),
                sap_pred,
                qs_pred,
            )
        return {
            "prediction": prediction,
            "sap_prediction": sap_pred,
            "qs_prediction": qs_pred,
            "pooled_embedding": features.pooled_embedding,
        }


class MultiTaskSpeechRegressor(nn.Module):
    def __init__(
        self,
        backbone: HFSSLBackbone,
        task_ids: Iterable[str],
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        shared_hidden_dim = hidden_dim or backbone.hidden_size
        self.backbone = backbone
        self.ordered_task_ids = list(task_ids)
        self.heads = nn.ModuleDict(
            {
                task_id: RegressionHead(backbone.hidden_size, shared_hidden_dim, dropout=dropout)
                for task_id in self.ordered_task_ids
            }
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        task_ids: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(input_values=input_values, attention_mask=attention_mask)
        if task_ids is None:
            raise ValueError("task_ids are required for multi-task forward passes")
        if list(task_ids) != self.ordered_task_ids:
            raise ValueError("task_ids must match the model's ordered_task_ids")
        prediction_matrix = torch.stack(
            [self.heads[task_id](features.pooled_embedding) for task_id in self.ordered_task_ids],
            dim=1,
        )
        return {
            "prediction": prediction_matrix,
            "pooled_embedding": features.pooled_embedding,
            "task_ids": self.ordered_task_ids,
        }
