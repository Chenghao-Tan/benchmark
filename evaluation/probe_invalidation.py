from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from utils.registry import register


@register("probe_invalidation")
class ProbeInvalidationEvaluation(EvaluationObject):
    def __init__(
        self,
        target_model=None,
        noise_variance: float = 0.01,
        n_samples: int = 10000,
        target_class: int | str | None = 1,
        **kwargs,
    ):
        if target_model is None:
            raise ValueError("ProbeInvalidationEvaluation requires target_model")
        self._target_model = target_model
        self._noise_variance = float(noise_variance)
        self._n_samples = int(n_samples)
        self._target_class = target_class

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
            return pd.DataFrame([{"average_invalidation_rate": float("nan")}])

        counterfactual_selected = counterfactual_features.loc[
            selected_mask.to_numpy()
        ].copy(deep=True)
        class_to_index = self._target_model.get_class_to_index()
        if self._target_class is None:
            desired_index = max(class_to_index.values())
        else:
            desired_index = class_to_index[self._target_class]

        invalidation_rates = []
        std = float(np.sqrt(self._noise_variance))
        for _, row in counterfactual_selected.iterrows():
            base = row.to_numpy(dtype=np.float32)
            perturbed = np.repeat(base.reshape(1, -1), self._n_samples, axis=0)
            perturbed += np.random.normal(
                loc=0.0,
                scale=std,
                size=perturbed.shape,
            ).astype(np.float32)
            perturbed_df = pd.DataFrame(perturbed, columns=counterfactual_selected.columns)
            prediction = self._target_model.get_prediction(perturbed_df, proba=False)
            labels = prediction.argmax(dim=1).detach().cpu().numpy()
            invalidation_rates.append(float(np.mean(labels != desired_index)))

        return pd.DataFrame(
            [{"average_invalidation_rate": float(np.mean(invalidation_rates))}]
        )
