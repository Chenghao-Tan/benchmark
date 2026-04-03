"""Random forest target model implementation."""

from __future__ import annotations

import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from dataset.dataset_object import DatasetObject
from model.model_object import ModelObject, process_nan
from utils.registry import register
from utils.seed import seed_context


@register("randomforest")
class RandomForestModel(ModelObject):
    """Train a scikit-learn random forest classifier for benchmark datasets.

    Args:
        seed: Seed passed to the underlying random forest and seeding context.
        device: Included for API compatibility with torch-based models.
        n_estimators: Number of trees in the ensemble.
        max_depth: Optional maximum tree depth.
        min_samples_split: Minimum samples required to split an internal node.
    """

    def __init__(
        self,
        seed: int | None = None,
        device: str = "cpu",
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        **kwargs,
    ):
        self._model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=seed,
        )
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False

    def fit(self, trainset: DatasetObject | None):
        """Train the random forest on a finalized dataset.

        Args:
            trainset: Frozen training dataset.

        Raises:
            ValueError: If ``trainset`` is missing.
        """
        if trainset is None:
            raise ValueError("trainset is required for RandomForestModel.fit()")
        with seed_context(self._seed):
            X, labels, _ = self.extract_training_data(trainset)
            self._model.fit(X, labels.cpu().numpy())
            self._is_trained = True

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        """Predict probabilities or hard labels for feature rows.

        Args:
            X: Feature matrix without the target column.
            proba: When ``True``, return probabilities. Otherwise return one-hot
                hard predictions.

        Returns:
            torch.Tensor: Prediction tensor on the CPU.
        """
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        with seed_context(self._seed):
            probabilities = torch.tensor(
                self._model.predict_proba(X), dtype=torch.float32
            )
            if proba:
                return probabilities
            indices = probabilities.argmax(dim=1)
            return torch.nn.functional.one_hot(
                indices, num_classes=probabilities.shape[1]
            ).to(dtype=torch.float32)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Reject torch forward calls for the non-torch random forest model.

        Raises:
            TypeError: Always raised because the underlying estimator is not
                differentiable through torch.
        """
        with seed_context(self._seed):
            raise TypeError(
                "RandomForestModel.forward() is unavailable because the underlying model is not torch-based"
            )
