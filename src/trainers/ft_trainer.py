from __future__ import annotations

from typing import Any

import pandas as pd

from src.tasks.pair_builder import sample_auxiliary_frame
from src.trainers.base import BaseTrainer


class FTTrainer(BaseTrainer):
    def run(
        self,
        qs_train: pd.DataFrame,
        qs_val: pd.DataFrame,
        sap_train: pd.DataFrame,
        sap_val: pd.DataFrame,
        sap_test: pd.DataFrame,
    ) -> dict[str, Any]:
        ratio = float(self.config["experiment"].get("ratio", 1.0))
        effective_n = int(round(ratio * len(qs_train)))
        sampled_aux, aux_meta = sample_auxiliary_frame(qs_train, "label_raw", effective_n, self.config["experiment"]["seed"])
        stage1_train = sampled_aux.assign(label_for_loss=sampled_aux["label_raw"], domain="qualispeech")
        stage1_val = qs_val.assign(label_for_loss=qs_val["label_raw"], domain="qualispeech")
        model = self.build_single_head_model()
        stage1 = self.run_stage(
            "stage1",
            model,
            stage1_train,
            stage1_val,
            mode="single",
            stage_cfg=self.config.get("ft", {}).get("stage1", self.config["training"]),
        )

        unwrapped = self._unwrap(model)
        self._reset_head(unwrapped, self.config.get("ft", {}).get("head_reset", "reuse_full_head"))
        self._apply_freeze_schedule(unwrapped, self.config.get("ft", {}).get("freeze_schedule", "full_finetune"))
        stage2_train = sap_train.assign(label_for_loss=sap_train["label"], domain="sap")
        stage2_val = sap_val.assign(label_for_loss=sap_val["label"], domain="sap")
        stage2 = self.run_stage(
            "stage2",
            model,
            stage2_train,
            stage2_val,
            mode="single",
            stage_cfg=self.config.get("ft", {}).get("stage2", self.config["training"]),
            resume=False,
        )
        metrics = self.finalize_run(
            model,
            test_frame=sap_test,
            mode="single",
            run_metadata={**self.config["experiment"], **aux_meta},
            copy_stage_checkpoint_from=stage2.best_checkpoint,
        )
        return {"stage1": stage1, "stage2": stage2, "metrics": metrics, "auxiliary": aux_meta}

