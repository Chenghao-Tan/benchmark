from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from utils.registry import register


@register("claproar_credit_stats")
class ClaproarCreditStatsEvaluation(EvaluationObject):
    def __init__(self, target_model=None, **kwargs):
        self._target_model = target_model

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        (
            factual_features,
            counterfactual_features,
            evaluation_mask,
            success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)
        selected_mask = evaluation_mask & success_mask
        if int(selected_mask.sum()) == 0:
            return pd.DataFrame(
                [{"feature_change_std_mean": float("nan"), "individual_cost": float("nan")}]
            )

        factual_selected = factual_features.loc[selected_mask.to_numpy()].copy(deep=True)
        counterfactual_selected = counterfactual_features.loc[
            selected_mask.to_numpy()
        ].copy(deep=True)
        difference = np.abs(
            counterfactual_selected.to_numpy(dtype=np.float32)
            - factual_selected.to_numpy(dtype=np.float32)
        )
        feature_change_std_mean = float(np.std(difference, axis=0).mean())
        individual_cost = float(np.linalg.norm(difference, axis=1).mean())
        return pd.DataFrame(
            [
                {
                    "feature_change_std_mean": feature_change_std_mean,
                    "individual_cost": individual_cost,
                }
            ]
        )
