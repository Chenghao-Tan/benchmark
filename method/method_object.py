"""Base abstraction for counterfactual generation methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from dataset.dataset_object import DatasetObject
from model.model_object import ModelObject


class MethodObject(ABC):
    """Define the interface shared by all counterfactual generation methods.

    Implementations wrap a trained target model, optionally fit any method-
    specific state, and generate counterfactual feature rows that align with the
    benchmark dataset format.
    """

    _target_model: ModelObject
    _seed: int | None = None
    _device: str
    _need_grad: bool
    _is_trained: bool = False
    _desired_class: int | str | None = None

    @abstractmethod
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        **kwargs,
    ):
        """Initialize a counterfactual generation method.

        Args:
            target_model: Trained target model used to score or guide
                counterfactual generation.
            seed: Seed used for deterministic method behavior.
            device: Execution device, typically ``"cpu"`` or ``"cuda"``.
            desired_class: Optional target class the method should steer
                counterfactuals toward.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, trainset: DatasetObject | None):
        """Fit any method-specific state required before generation.

        Args:
            trainset: Frozen training dataset, or ``None`` for methods that do
                not require fitting.
        """
        raise NotImplementedError

    @abstractmethod
    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        """Generate counterfactual features for a batch of factual inputs.

        Args:
            factuals: Feature-only DataFrame for the current batch.

        Returns:
            pd.DataFrame: Counterfactual feature rows with the same row count
            and feature columns as ``factuals``.
        """
        raise NotImplementedError

    def predict(self, testset: DatasetObject, batch_size: int = 20) -> DatasetObject:
        """Generate a frozen counterfactual dataset from a factual dataset.

        Args:
            testset: Frozen factual dataset to transform.
            batch_size: Number of rows processed per generation batch.

        Returns:
            DatasetObject: Frozen clone of ``testset`` marked as a
            counterfactual dataset.

        Raises:
            RuntimeError: If the method has not been trained yet.
            ValueError: If the input dataset is invalid or a generated batch
                does not preserve the required row or column layout.
        """
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if getattr(testset, "counterfactual", False):
            raise ValueError("testset must not already be marked as counterfactual")

        factuals = testset.get(target=False)
        counterfactual_batches: list[pd.DataFrame] = []

        for start in range(0, factuals.shape[0], batch_size):
            batch = factuals.iloc[start : start + batch_size]
            counterfactual_batch = self.get_counterfactuals(batch)

            if counterfactual_batch.shape[0] != batch.shape[0]:
                raise ValueError(
                    "get_counterfactuals() must preserve the input row count"
                )
            if set(counterfactual_batch.columns) != set(batch.columns):
                raise ValueError(
                    "get_counterfactuals() must preserve the input feature columns"
                )

            counterfactual_batch = counterfactual_batch.reindex(
                index=batch.index, columns=batch.columns
            )
            counterfactual_batches.append(counterfactual_batch)

        if counterfactual_batches:
            counterfactual_features = pd.concat(counterfactual_batches, axis=0)
            counterfactual_features = counterfactual_features.reindex(
                index=factuals.index
            )
        else:
            counterfactual_features = factuals.iloc[0:0].copy(deep=True)

        target_column = testset.target_column
        counterfactual_target = pd.DataFrame(
            -1.0,
            index=counterfactual_features.index,
            columns=[target_column],
        )
        counterfactual_df = pd.concat(
            [counterfactual_features, counterfactual_target], axis=1
        )
        counterfactual_df = counterfactual_df.reindex(
            columns=testset.ordered_features()
        )

        output = testset.clone()
        output.update("counterfactual", True, df=counterfactual_df)

        if self._desired_class is not None:
            class_to_index = self._target_model.get_class_to_index()
            prediction = self._target_model.predict(testset, batch_size=batch_size)
            predicted_label = prediction.argmax(dim=1).cpu().numpy()
            evaluation_filter = pd.DataFrame(
                predicted_label != class_to_index[self._desired_class],
                index=counterfactual_df.index,
                columns=["evaluation_filter"],
                dtype=bool,
            )
            output.update("evaluation_filter", evaluation_filter)

        output.freeze()
        return output
