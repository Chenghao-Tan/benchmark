from __future__ import annotations

from abc import ABC, abstractmethod

from dataset.dataset_object import DatasetObject


class PreProcessObject(ABC):
    _seed: int | None = None

    @abstractmethod
    def __init__(self, seed: int | None = None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def transform(
        self, input: DatasetObject
    ) -> DatasetObject | tuple[DatasetObject, ...]:
        raise NotImplementedError
