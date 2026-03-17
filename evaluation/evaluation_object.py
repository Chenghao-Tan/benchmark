from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from dataset.dataset_object import DatasetObject


class EvaluationObject(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        raise NotImplementedError
