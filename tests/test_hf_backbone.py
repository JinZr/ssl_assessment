from __future__ import annotations

from types import SimpleNamespace

import torch

from src.models.hf_ssl_backbone import HFSSLBackbone


class FakeConfig:
    hidden_size = 4
    conv_kernel = [3]
    conv_stride = [2]

    def to_dict(self) -> dict[str, object]:
        return {"hidden_size": 4, "conv_kernel": [3], "conv_stride": [2]}


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = FakeConfig()

    def gradient_checkpointing_enable(self) -> None:
        return None

    def _get_feature_vector_attention_mask(self, output_length: int, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask[:, ::2]
        return mask[:, :output_length].bool()

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None, output_hidden_states: bool = False):
        hidden = input_values[:, ::2].unsqueeze(-1).repeat(1, 1, 4)
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=None)


class FakeProcessor:
    def __call__(self, waveforms, sampling_rate: int, padding: bool, return_tensors: str):
        waveforms = [torch.as_tensor(waveform, dtype=torch.float32) for waveform in waveforms]
        max_len = max(waveform.shape[0] for waveform in waveforms)
        batch = torch.zeros(len(waveforms), max_len)
        mask = torch.zeros(len(waveforms), max_len, dtype=torch.long)
        for index, waveform in enumerate(waveforms):
            batch[index, : waveform.shape[0]] = waveform
            mask[index, : waveform.shape[0]] = 1
        return {"input_values": batch, "attention_mask": mask}


def test_hf_ssl_backbone_uses_masked_mean_pool(monkeypatch) -> None:
    from src.models import hf_ssl_backbone as module

    monkeypatch.setattr(module.AutoConfig, "from_pretrained", lambda *args, **kwargs: FakeConfig())
    monkeypatch.setattr(module.AutoModel, "from_pretrained", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: FakeProcessor())

    backbone = HFSSLBackbone("wavlm_base")
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    outputs = backbone(inputs, attention_mask=mask)
    assert outputs.last_hidden_state.shape == (1, 2, 4)
    assert torch.allclose(outputs.pooled_embedding, torch.tensor([[2.0, 2.0, 2.0, 2.0]]))
