"""Validity evaluation for counterfactual outputs."""

from __future__ import annotations

import pandas as pd

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from utils.registry import register


@register("validity")
class ValidityEvaluation(EvaluationObject):
    """Measure the fraction of evaluated counterfactuals that are successful."""

    def __init__(self, **kwargs):
        pass

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        """Compute validity over the rows selected for evaluation.

        Args:
            factuals: Frozen dataset containing the original inputs.
            counterfactuals: Frozen dataset containing generated
                counterfactuals.

        Returns:
            pd.DataFrame: Single-row table with the ``validity`` metric.
        """
        (
            factual_features,
            counterfactual_features,
            evaluation_mask,
            success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)

        evaluated_success_mask = success_mask.loc[evaluation_mask.to_numpy()]
        if evaluated_success_mask.shape[0] == 0:
            validity = float("nan")
        else:
            validity = float(evaluated_success_mask.astype(float).mean())

        return pd.DataFrame([{"validity": validity}])
