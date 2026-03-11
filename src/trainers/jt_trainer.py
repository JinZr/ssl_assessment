from __future__ import annotations

from typing import Any

import pandas as pd

from src.tasks.pair_builder import sample_auxiliary_frame
from src.trainers.base import BaseTrainer


class JTTrainer(BaseTrainer):
    def run(
        self,
        sap_train: pd.DataFrame,
        sap_val: pd.DataFrame,
        sap_test: pd.DataFrame,
        qs_train: pd.DataFrame,
        ratio_mode: str = "jt",
        dual_head: bool = False,
    ) -> dict[str, Any]:
        ratio = float(self.config["experiment"].get("ratio", 1.0))
        if ratio_mode != "jt":
            raise ValueError("JTTrainer only supports JT auxiliary ratio semantics.")
        effective_n = int(round(ratio * len(sap_train)))
        sampled_aux, aux_meta = sample_auxiliary_frame(qs_train, "label_raw", effective_n, self.config["experiment"]["seed"])
        combined = pd.concat(
            [
                sap_train.assign(label_for_loss=sap_train["label"], domain="sap"),
                sampled_aux.assign(
                    label_for_loss=sampled_aux["label_raw"] if dual_head else sampled_aux["label_aligned"],
                    domain="qualispeech",
                ),
            ],
            ignore_index=True,
        ).sample(frac=1.0, random_state=self.config["experiment"]["seed"])
        model = self.build_dual_head_model() if dual_head else self.build_single_head_model()
        mode = "dual" if dual_head else "single"
        stage = self.run_stage("train", model, combined, sap_val, mode=mode)
        metrics = self.finalize_run(
            model,
            test_frame=sap_test,
            mode=mode,
            run_metadata={**self.config["experiment"], **aux_meta},
            copy_stage_checkpoint_from=stage.best_checkpoint,
        )
        return {"stage": stage, "metrics": metrics, "auxiliary": aux_meta}

