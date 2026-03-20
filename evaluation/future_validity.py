from __future__ import annotations

import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from utils.registry import register


@register("future_validity")
class FutureValidityEvaluation(EvaluationObject):
    def __init__(self, target_model=None, target_class: int | str | None = 1, **kwargs):
        if target_model is None:
            raise ValueError("FutureValidityEvaluation requires target_model")
        self._target_model = target_model
        self._target_class = target_class

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        (
            _factual_features,
            counterfactual_features,
            evaluation_mask,
            success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)
        denominator = int(evaluation_mask.sum())
        if denominator == 0:
            return pd.DataFrame([{"future_validity": float("nan")}])

        selected_mask = evaluation_mask & success_mask
        if int(selected_mask.sum()) == 0:
            return pd.DataFrame([{"future_validity": 0.0}])

        counterfactual_selected = counterfactual_features.loc[
            selected_mask.to_numpy()
        ].copy(deep=True)
        prediction = self._target_model.get_prediction(
            counterfactual_selected,
            proba=False,
        )
        class_to_index = self._target_model.get_class_to_index()
        if self._target_class is None:
            desired_index = max(class_to_index.values())
        else:
            desired_index = class_to_index[self._target_class]
        labels = prediction.argmax(dim=1)
        future_validity = float(
            (labels == desired_index).to(dtype=torch.float32).sum().item() / denominator
        )
        return pd.DataFrame([{"future_validity": future_validity}])
