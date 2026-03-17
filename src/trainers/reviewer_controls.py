from __future__ import annotations

from typing import Any

import pandas as pd

from src.trainers.base import BaseTrainer
from src.trainers.jt_trainer import JTTrainer


class DualHeadJTTrainer(JTTrainer):
    def run(
        self,
        sap_train: pd.DataFrame,
        sap_val: pd.DataFrame,
        sap_test: pd.DataFrame,
        qs_train: pd.DataFrame,
    ) -> dict[str, Any]:
        return super().run(sap_train, sap_val, sap_test, qs_train, ratio_mode="jt", dual_head=True)


class SAPMultiTaskTrainer(BaseTrainer):
    def run(
        self,
        multitask_train: pd.DataFrame,
        multitask_val: pd.DataFrame,
        multitask_test: pd.DataFrame,
    ) -> dict[str, Any]:
        task_ids = list(self.config["experiment"]["multitask_tasks"])
        stage_cfg = self.config["training"]
        model = self.build_multi_task_model(task_ids=task_ids)
        stage = self.run_stage("train", model, multitask_train, multitask_val, mode="multitask", stage_cfg=stage_cfg)
        metrics = self.finalize_run(
            model,
            test_frame=multitask_test,
            mode="multitask",
            run_metadata=self.config["experiment"],
            copy_stage_checkpoint_from=stage.best_checkpoint,
            stage_cfg=stage_cfg,
        )
        return {"stage": stage, "metrics": metrics}
