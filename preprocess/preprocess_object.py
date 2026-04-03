"""Base abstraction for preprocessing steps in the benchmark pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dataset.dataset_object import DatasetObject


class PreProcessObject(ABC):
    """Define the interface shared by all preprocessing components."""

    _seed: int | None = None

    @abstractmethod
    def __init__(self, seed: int | None = None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def transform(
        self, input: DatasetObject
    ) -> DatasetObject | tuple[DatasetObject, ...]:
        """Transform a mutable dataset into one or more downstream datasets.

        Args:
            input: Mutable dataset instance to transform.

        Returns:
            DatasetObject | tuple[DatasetObject, ...]: Transformed dataset or
            datasets ready for the next pipeline stage.
        """
        raise NotImplementedError
