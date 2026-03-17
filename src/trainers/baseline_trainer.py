from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.trainers.base import BaseTrainer


class BaselineTrainer(BaseTrainer):
    def run(
        self,
        train_frame: pd.DataFrame,
        val_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        stage_cfg = self.config["training"]
        model = self.build_single_head_model()
        stage = self.run_stage("train", model, train_frame, val_frame, mode="single", stage_cfg=stage_cfg)
        metrics = self.finalize_run(
            model,
            test_frame=test_frame,
            mode="single",
            run_metadata=self.config["experiment"],
            copy_stage_checkpoint_from=stage.best_checkpoint,
            stage_cfg=stage_cfg,
        )
        return {"stage": stage, "metrics": metrics}
