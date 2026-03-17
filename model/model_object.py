from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps

import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from dataset.dataset_object import DatasetObject


def process_nan():
    def decorator(func):
        @wraps(func)
        def wrapper(self, X: pd.DataFrame, *args, **kwargs):
            X_work = X.copy(deep=True)
            nan_rows = X_work.isna().any(axis=1)
            if nan_rows.any():
                X_work.loc[nan_rows, :] = 0.0
            y = func(self, X_work, *args, **kwargs)
            if nan_rows.any():
                y = y.clone()
                y[nan_rows.to_numpy()] = -1
            return y

        return wrapper

    return decorator


class ModelObject(ABC):
    _model: torch.nn.Module | RandomForestClassifier
    _seed: int | None = None
    _device: str
    _need_grad: bool
    _is_trained: bool = False
    _class_to_index: dict[int | str, int] | None = None

    @abstractmethod
    def __init__(self, seed: int | None = None, device: str = "cpu", **kwargs):
        raise NotImplementedError

    @abstractmethod
    def fit(self, trainset: DatasetObject | None):
        raise NotImplementedError

    @abstractmethod
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, testset: DatasetObject, batch_size: int = 20) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        X = testset.get(target=False)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(X), batch_size):
            batch = X.iloc[start : start + batch_size]
            outputs.append(self.get_prediction(batch, proba=False).detach().cpu())
        return torch.cat(outputs, dim=0) if outputs else torch.empty(0)

    def predict_proba(
        self, testset: DatasetObject, batch_size: int = 20
    ) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        X = testset.get(target=False)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(X), batch_size):
            batch = X.iloc[start : start + batch_size]
            outputs.append(self.get_prediction(batch, proba=True).detach().cpu())
        return torch.cat(outputs, dim=0) if outputs else torch.empty(0)

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)

    def extract_training_data(
        self,
        trainset: DatasetObject,
    ) -> tuple[pd.DataFrame, torch.Tensor, int]:
        X = trainset.get(target=False)
        y = trainset.get(target=True)

        if y.shape[1] == 1:
            target = y.iloc[:, 0]
            if target.isna().any():
                raise ValueError("Target labels cannot contain NaN")

            unique_values = list(pd.Index(target.unique()))
            if len(unique_values) == 0:
                raise ValueError("Target labels cannot be empty")

            if all(isinstance(value, str) for value in unique_values):
                sorted_values = sorted(unique_values)
            elif all(isinstance(value, int) for value in unique_values):
                target = target.map(int)
                sorted_values = sorted(pd.Index(target.unique()).tolist())
            else:
                raise TypeError(
                    "Single target_column must contain either all string or all integer labels"
                )

            self._class_to_index = {
                class_value: index for index, class_value in enumerate(sorted_values)
            }
            labels = torch.tensor(
                [self._class_to_index[value] for value in target.tolist()],
                dtype=torch.long,
            )
            output_dim = max(2, len(self._class_to_index))
            return X, labels, output_dim
        else:
            labels = torch.tensor(y.to_numpy().argmax(axis=1), dtype=torch.long)
            output_dim = y.shape[1]
            self._class_to_index = {index: index for index in range(output_dim)}
            return X, labels, output_dim

    def get_class_to_index(self) -> dict[int | str, int]:
        if self._class_to_index is None:
            raise RuntimeError("Target model class mapping is unavailable")
        return dict(self._class_to_index)
