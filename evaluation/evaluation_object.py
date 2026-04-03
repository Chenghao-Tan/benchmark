"""Base abstraction for counterfactual evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from dataset.dataset_object import DatasetObject


class EvaluationObject(ABC):
    """Define the interface for metrics computed on factual/counterfactual pairs."""

    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        """Compute one or more metrics for aligned factual and counterfactual sets.

        Args:
            factuals: Frozen dataset containing the original inputs.
            counterfactuals: Frozen dataset containing generated counterfactuals.

        Returns:
            pd.DataFrame: Single-row table with metric names as columns.
        """
        raise NotImplementedError
