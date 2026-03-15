from __future__ import annotations

import math
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data.datasets import ManifestDataset, SpeechCollator
from src.eval.evaluate import build_prediction_frame, metric_payload
from src.models.hf_ssl_backbone import HFSSLBackbone
from src.models.regression import DualHeadSpeechRegressor, MultiTaskSpeechRegressor, SpeechRegressor
from src.samplers.dynamic_batch import DynamicDurationBatchSampler
from src.utils.distributed import cleanup_distributed, init_distributed, is_distributed, is_main_process
from src.utils.io import write_csv, write_json
from src.utils.metrics import compute_metrics


@dataclass
class StageResult:
    stage_name: str
    best_metrics: dict[str, float]
    best_checkpoint: str
    last_checkpoint: str


class BaseTrainer:
    def __init__(self, config: dict[str, Any], run_dir: str | Path) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rank, self.world_size, self.local_rank = init_distributed()
        self.device = self._resolve_device()
        self.loss_fn = nn.MSELoss()
        self.train_history: list[dict[str, Any]] = []
        self.val_history: list[dict[str, Any]] = []
        self._precision_enabled, self._precision_dtype = self._resolve_precision()

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(device)
            return device
        return torch.device("cpu")

    def _resolve_precision(self) -> tuple[bool, torch.dtype | None]:
        precision = self.config.get("training", {}).get("precision", "auto")
        if self.device.type != "cuda":
            return False, None
        if precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
            return True, torch.bfloat16
        if precision == "fp16" or precision == "auto":
            return True, torch.float16
        return False, None

    def _autocast(self):
        if not self._precision_enabled or self._precision_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self._precision_dtype)

    def cleanup(self) -> None:
        cleanup_distributed()

    def _unwrap(self, model: nn.Module) -> nn.Module:
        if isinstance(model, DistributedDataParallel):
            return model.module
        return model

    def build_backbone(self) -> HFSSLBackbone:
        model_cfg = self.config["model"]
        return HFSSLBackbone(
            model_name=model_cfg["name"],
            cache_dir=model_cfg.get("cache_dir"),
            revision=model_cfg.get("revision"),
            apply_spec_augment=model_cfg.get("apply_spec_augment", False),
            layerdrop=model_cfg.get("layerdrop", 0.0),
            output_hidden_states=model_cfg.get("output_hidden_states", False),
            gradient_checkpointing=model_cfg.get("gradient_checkpointing"),
        )

    def build_single_head_model(self) -> SpeechRegressor:
        backbone = self.build_backbone()
        return SpeechRegressor(
            backbone=backbone,
            hidden_dim=self.config["model"].get("head_hidden_dim"),
            dropout=self.config["model"].get("dropout", 0.1),
        )

    def build_dual_head_model(self) -> DualHeadSpeechRegressor:
        backbone = self.build_backbone()
        return DualHeadSpeechRegressor(
            backbone=backbone,
            hidden_dim=self.config["model"].get("head_hidden_dim"),
            dropout=self.config["model"].get("dropout", 0.1),
        )

    def build_multi_task_model(self, task_ids: list[str]) -> MultiTaskSpeechRegressor:
        backbone = self.build_backbone()
        return MultiTaskSpeechRegressor(
            backbone=backbone,
            task_ids=task_ids,
            hidden_dim=self.config["model"].get("head_hidden_dim"),
            dropout=self.config["model"].get("dropout", 0.1),
        )

    def maybe_wrap_ddp(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        if is_distributed():
            model = DistributedDataParallel(
                model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            )
        return model

    def _resolve_max_total_sec(self, stage_cfg: dict[str, Any]) -> float:
        experiment_override = self.config.get("experiment", {}).get("max_total_sec_override")
        if experiment_override is not None:
            return float(experiment_override)
        if stage_cfg is not self.config.get("training") and stage_cfg.get("max_total_sec") is not None:
            return float(stage_cfg["max_total_sec"])
        if self.config["model"].get("max_total_sec") is not None:
            return float(self.config["model"]["max_total_sec"])
        if self.config.get("training", {}).get("max_total_sec") is not None:
            return float(self.config["training"]["max_total_sec"])
        return 180.0

    def _resolve_max_input_sec(self, stage_cfg: dict[str, Any]) -> float:
        experiment_override = self.config.get("experiment", {}).get("max_input_sec_override")
        if experiment_override is not None:
            return float(experiment_override)
        if stage_cfg is not self.config.get("training") and stage_cfg.get("max_input_sec") is not None:
            return float(stage_cfg["max_input_sec"])
        if self.config["model"].get("max_input_sec") is not None:
            return float(self.config["model"]["max_input_sec"])
        if self.config.get("training", {}).get("max_input_sec") is not None:
            return float(self.config["training"]["max_input_sec"])
        return self._resolve_max_total_sec(stage_cfg)

    def _loss_name(self, stage_cfg: dict[str, Any]) -> str:
        return stage_cfg.get("loss", self.config.get("training", {}).get("loss", "mse")).lower()

    def _huber_delta(self, stage_cfg: dict[str, Any]) -> float:
        return float(stage_cfg.get("huber_delta", self.config.get("training", {}).get("huber_delta", 1.0)))

    def _elementwise_loss(self, preds: torch.Tensor, targets: torch.Tensor, stage_cfg: dict[str, Any]) -> torch.Tensor:
        loss_name = self._loss_name(stage_cfg)
        if loss_name == "mse":
            return (preds - targets) ** 2
        if loss_name == "huber":
            return nn.functional.huber_loss(preds, targets, delta=self._huber_delta(stage_cfg), reduction="none")
        raise ValueError(f"Unsupported loss: {loss_name}")

    def _make_loader(
        self,
        frame: pd.DataFrame,
        processor: Any,
        train: bool,
        stage_cfg: dict[str, Any] | None = None,
        multitask_task_ids: list[str] | None = None,
    ) -> DataLoader:
        data_cfg = self.config.get("data", {})
        stage_cfg = stage_cfg or self.config.get("training", {})
        dataset = ManifestDataset(frame)
        durations = frame.get("duration_sec", pd.Series([1.0] * len(frame))).fillna(1.0).astype(float).tolist()
        max_total_sec = self._resolve_max_total_sec(stage_cfg)
        sampler = DynamicDurationBatchSampler(
            durations=durations,
            max_total_sec=max_total_sec,
            shuffle=train,
            seed=self.config["experiment"]["seed"],
            rank=self.rank if train else 0,
            world_size=self.world_size if train else 1,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=data_cfg.get("num_workers", 0),
            pin_memory=self.device.type == "cuda",
            collate_fn=SpeechCollator(
                processor=processor,
                sampling_rate=data_cfg.get("sample_rate", 16_000),
                multitask_task_ids=multitask_task_ids,
                max_input_sec=self._resolve_max_input_sec(stage_cfg),
            ),
        )

    def _build_optimizer(self, model: nn.Module, stage_cfg: dict[str, Any]) -> torch.optim.Optimizer:
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        return AdamW(
            trainable_params,
            lr=stage_cfg.get("lr", 1e-5),
            weight_decay=stage_cfg.get("weight_decay", 0.01),
            betas=tuple(stage_cfg.get("betas", [0.9, 0.999])),
            eps=stage_cfg.get("eps", 1e-8),
        )

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        stage_cfg: dict[str, Any],
        total_steps: int,
    ) -> LambdaLR | None:
        if stage_cfg.get("scheduler", "none") != "cosine_with_warmup":
            return None
        warmup_ratio = stage_cfg.get("warmup_ratio", 0.1)
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _forward(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        mode: str,
    ) -> torch.Tensor:
        input_values = batch["input_values"].to(self.device)
        attention_mask = batch["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if mode == "single":
            outputs = model(input_values=input_values, attention_mask=attention_mask)
        elif mode == "dual":
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                domains=batch.get("domains"),
            )
        elif mode == "multitask":
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                task_ids=batch.get("multitask_task_ids"),
            )
        else:
            raise ValueError(f"Unsupported forward mode: {mode}")
        return self._aggregate_segment_predictions(outputs["prediction"], batch)

    @staticmethod
    def _aggregate_segment_predictions(predictions: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        parent_indices = batch["segment_parent_indices"].to(predictions.device)
        weights = batch["segment_weights"].to(predictions.device)
        batch_size = len(batch["metadata"])
        if predictions.ndim == 1:
            aggregated = torch.zeros(batch_size, dtype=predictions.dtype, device=predictions.device)
            aggregated.index_add_(0, parent_indices, predictions * weights)
            return aggregated
        if predictions.ndim == 2:
            aggregated = torch.zeros(batch_size, predictions.shape[1], dtype=predictions.dtype, device=predictions.device)
            aggregated.index_add_(0, parent_indices, predictions * weights.unsqueeze(-1))
            return aggregated
        raise ValueError(f"Unsupported prediction rank for aggregation: {predictions.ndim}")

    def _compute_batch_loss(
        self,
        preds: torch.Tensor,
        batch: dict[str, Any],
        mode: str,
        stage_cfg: dict[str, Any],
    ) -> torch.Tensor:
        if mode == "multitask":
            task_labels = batch["task_labels"].to(self.device)
            task_mask = batch["task_mask"].to(self.device)
            losses = self._elementwise_loss(preds, task_labels, stage_cfg)
            masked = losses * task_mask.to(losses.dtype)
            return masked.sum() / task_mask.sum().clamp_min(1)
        labels = batch["labels"].to(self.device)
        losses = self._elementwise_loss(preds, labels, stage_cfg)
        return losses.mean()

    @staticmethod
    def _is_better(candidate: dict[str, float], best: dict[str, float] | None) -> bool:
        if best is None:
            return True
        candidate_key = (candidate["mse"], -candidate["lcc"], -candidate["srcc"])
        best_key = (best["mse"], -best["lcc"], -best["srcc"])
        return candidate_key < best_key

    def _save_checkpoint(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR | None,
        epoch: int,
        best_metrics: dict[str, float] | None,
    ) -> None:
        if not is_main_process(self.rank):
            return
        payload = {
            "model_state_dict": self._unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_metrics": best_metrics,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def _load_checkpoint(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR | None,
    ) -> tuple[int, dict[str, float] | None]:
        payload = torch.load(path, map_location=self.device)
        self._unwrap(model).load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler and payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        return int(payload.get("epoch", -1)) + 1, payload.get("best_metrics")

    def _reset_head(self, model: SpeechRegressor, reset_mode: str) -> None:
        if reset_mode == "reuse_full_head":
            return
        head = model.head
        layers = [module for module in head.network if isinstance(module, nn.Linear)]
        if reset_mode == "reset_last_linear_only":
            layers[-1].reset_parameters()
            return
        if reset_mode == "reset_full_head":
            for layer in layers:
                layer.reset_parameters()
            return
        raise ValueError(f"Unknown head reset mode: {reset_mode}")

    def _apply_freeze_schedule(self, model: SpeechRegressor, schedule: str) -> None:
        backbone = model.backbone.model
        for param in backbone.parameters():
            param.requires_grad = True
        if schedule == "full_finetune":
            return
        if schedule == "freeze_encoder":
            for param in backbone.parameters():
                param.requires_grad = False
            return
        if schedule.startswith("unfreeze_last_"):
            for param in backbone.parameters():
                param.requires_grad = False
            num_layers = int(schedule.replace("unfreeze_last_", "").replace("_layers", ""))
            encoder_layers = getattr(getattr(backbone, "encoder", None), "layers", None)
            if encoder_layers is None:
                return
            for layer in encoder_layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            return
        raise ValueError(f"Unknown freeze schedule: {schedule}")

    def _collect_predictions(self, model: nn.Module, frame: pd.DataFrame, mode: str) -> list[dict[str, Any]]:
        model.eval()
        multitask_task_ids = getattr(self._unwrap(model), "ordered_task_ids", None) if mode == "multitask" else None
        loader = self._make_loader(
            frame,
            self._unwrap(model).backbone.processor,
            train=False,
            stage_cfg=self.config.get("training", {}),
            multitask_task_ids=multitask_task_ids,
        )
        records: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                with self._autocast():
                    batch_preds = self._forward(model, batch, mode).detach().float().cpu()
                if mode == "multitask":
                    ordered_task_ids = list(getattr(self._unwrap(model), "ordered_task_ids"))
                    eval_task = batch["eval_task_ids"][0]
                    eval_index = ordered_task_ids.index(eval_task)
                    preds = batch_preds[:, eval_index].numpy().tolist()
                else:
                    preds = batch_preds.numpy().tolist()
                for meta, pred in zip(batch["metadata"], preds):
                    records.append(
                        {
                            "utt_id": meta.get("utt_id"),
                            "speaker_id": meta.get("speaker_id"),
                            "audio_path": meta.get("audio_path"),
                            "y_true": float(meta["label"]),
                            "y_pred": float(pred),
                            "domain": meta.get("domain", "sap"),
                            "target_dim": meta.get("target_dim", meta.get("task_dim")),
                            "label_min": float(meta.get("label_min", 1.0)),
                            "label_max": float(meta.get("label_max", 7.0)),
                            "prompt_category": meta.get("prompt_category"),
                            "etiology": meta.get("etiology"),
                            "variant": self.config.get("experiment", {}).get("variant"),
                        }
                    )
        return records

    def evaluate_frame(self, model: nn.Module, frame: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        prediction_frame = build_prediction_frame(
            self._collect_predictions(model, frame, mode),
            run_metadata={},
            clip_range=(
                float(frame["label_min"].min()),
                float(frame["label_max"].max()),
            ),
        )
        raw_metrics = compute_metrics(prediction_frame["y_true"], prediction_frame["y_pred"]).to_dict()
        clipped_metrics = compute_metrics(prediction_frame["y_true"], prediction_frame["y_pred_clipped"]).to_dict()
        metrics = {**raw_metrics, **{f"clipped_{key}": value for key, value in clipped_metrics.items()}}
        return prediction_frame, metrics

    def run_stage(
        self,
        stage_name: str,
        model: nn.Module,
        train_frame: pd.DataFrame,
        val_frame: pd.DataFrame,
        mode: str = "single",
        stage_cfg: dict[str, Any] | None = None,
        resume: bool = True,
    ) -> StageResult:
        stage_cfg = stage_cfg or self.config["training"]
        model = self.maybe_wrap_ddp(model)
        optimizer = self._build_optimizer(model, stage_cfg)
        multitask_task_ids = getattr(self._unwrap(model), "ordered_task_ids", None) if mode == "multitask" else None
        train_loader = self._make_loader(
            train_frame,
            self._unwrap(model).backbone.processor,
            train=True,
            stage_cfg=stage_cfg,
            multitask_task_ids=multitask_task_ids,
        )
        total_steps = max(1, len(train_loader) * stage_cfg.get("max_epochs", 30))
        scheduler = self._build_scheduler(optimizer, stage_cfg, total_steps)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and self._precision_dtype == torch.float16,
        )

        stage_dir = self.run_dir / stage_name
        best_ckpt = stage_dir / "best.ckpt"
        last_ckpt = stage_dir / "last.ckpt"
        start_epoch = 0
        best_metrics: dict[str, float] | None = None
        patience = stage_cfg.get("patience", 5)
        patience_counter = 0
        if resume and last_ckpt.exists():
            start_epoch, best_metrics = self._load_checkpoint(last_ckpt, model, optimizer, scheduler)

        for epoch in range(start_epoch, stage_cfg.get("max_epochs", 30)):
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            num_batches = 0
            optimizer.zero_grad(set_to_none=True)
            accumulation_steps = stage_cfg.get("gradient_accumulation_steps", 1)
            for step, batch in enumerate(train_loader, start=1):
                with self._autocast():
                    preds = self._forward(model, batch, mode)
                    loss = self._compute_batch_loss(preds, batch, mode, stage_cfg) / accumulation_steps
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if step % accumulation_steps == 0 or step == len(train_loader):
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                running_loss += float(loss.detach().cpu()) * accumulation_steps
                num_batches += 1

            _, val_metrics = self.evaluate_frame(model, val_frame, mode)
            train_record = {
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": running_loss / max(1, num_batches),
            }
            val_record = {"stage": stage_name, "epoch": epoch, **val_metrics}
            self.train_history.append(train_record)
            self.val_history.append(val_record)

            self._save_checkpoint(last_ckpt, model, optimizer, scheduler, epoch, best_metrics)
            if self._is_better(val_metrics, best_metrics):
                best_metrics = {key: float(value) for key, value in val_metrics.items() if isinstance(value, (int, float))}
                self._save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch, best_metrics)
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

        if best_ckpt.exists():
            if dist.is_initialized():
                dist.barrier()
            payload = torch.load(best_ckpt, map_location=self.device)
            self._unwrap(model).load_state_dict(payload["model_state_dict"])
        if is_main_process(self.rank):
            write_csv(self.run_dir / "train_log.csv", pd.DataFrame(self.train_history))
            write_csv(self.run_dir / "val_metrics.csv", pd.DataFrame(self.val_history))
        return StageResult(
            stage_name=stage_name,
            best_metrics=best_metrics or {},
            best_checkpoint=str(best_ckpt),
            last_checkpoint=str(last_ckpt),
        )

    def finalize_run(
        self,
        model: nn.Module,
        test_frame: pd.DataFrame,
        mode: str,
        run_metadata: dict[str, Any],
        copy_stage_checkpoint_from: str | None = None,
    ) -> dict[str, Any]:
        prediction_records = self._collect_predictions(model, test_frame, mode)
        prediction_frame = build_prediction_frame(
            prediction_records,
            run_metadata=run_metadata,
            clip_range=(float(test_frame["label_min"].min()), float(test_frame["label_max"].max())),
        )
        metrics = metric_payload(
            prediction_frame,
            n_bootstrap=self.config.get("evaluation", {}).get("n_bootstrap", 10_000),
            seed=self.config["experiment"]["seed"],
        )
        if is_main_process(self.rank):
            write_csv(self.run_dir / "test_predictions.csv", prediction_frame)
            write_json(self.run_dir / "test_metrics.json", metrics)
            write_json(
                self.run_dir / "model_info.json",
                {
                    "model_name": self.config["model"].get("name"),
                    "model_id": self.config["model"].get("model_id"),
                    "requested_revision": self.config["model"].get("requested_revision", self.config["model"].get("revision")),
                    "resolved_revision": self.config["model"].get("resolved_revision"),
                    "resolved_config": getattr(self._unwrap(model).backbone, "config_dict", {}),
                    "world_size": self.world_size,
                    "effective_batch_size": self.world_size * self.config["training"].get("gradient_accumulation_steps", 1),
                    "accumulation_steps": self.config["training"].get("gradient_accumulation_steps", 1),
                    "max_total_sec": self._resolve_max_total_sec(self.config.get("training", {})),
                    "max_input_sec": self._resolve_max_input_sec(self.config.get("training", {})),
                },
            )
            if copy_stage_checkpoint_from:
                source_dir = Path(copy_stage_checkpoint_from).parent
                shutil.copy2(source_dir / "best.ckpt", self.run_dir / "best.ckpt")
                shutil.copy2(source_dir / "last.ckpt", self.run_dir / "last.ckpt")
        return metrics
