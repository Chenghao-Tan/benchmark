from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml


class DatasetObject(ABC):
    _rawdf: pd.DataFrame
    _freeze: bool = False
    target_column: str
    raw_feature_type: dict[str, str]
    raw_feature_mutability: dict[str, bool]
    raw_feature_actionability: dict[str, str]

    @abstractmethod
    def __init__(self, path: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _read_df(self, path: str) -> pd.DataFrame:
        raise NotImplementedError

    def _read_attrs(self, path: str) -> dict[str, object]:
        attrs_path = Path(path) / f"{Path(path).name}.yaml"
        with attrs_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _ensure_mutable(self) -> None:
        if self._freeze:
            raise RuntimeError("Dataset is frozen; snapshot()/update() are unavailable")

    def _ensure_frozen(self) -> None:
        if not self._freeze:
            raise RuntimeError(
                "Dataset is mutable; get()/ordered_features()/__len__()/__getitem__() are unavailable"
            )

    def snapshot(self) -> pd.DataFrame:
        self._ensure_mutable()
        return self._rawdf.copy(deep=True)

    def update(self, flag: str, value: object, df: pd.DataFrame | None = None) -> bool:
        self._ensure_mutable()
        if flag is None:
            raise ValueError("flag must not be None")
        if df is not None:
            self._rawdf = df.copy(deep=True)
        setattr(self, flag, deepcopy(value))
        return True

    def attr(self, flag: str) -> object:
        if flag.startswith("_"):
            raise AttributeError(f"Access to protected member '{flag}' is forbidden")
        if not hasattr(self, flag):
            raise AttributeError(f"Unknown dataset attribute: {flag}")
        return deepcopy(getattr(self, flag))

    def freeze(self):
        self._freeze = True

    def get(self, target: bool = False) -> pd.DataFrame:
        self._ensure_frozen()
        target_column = self.target_column
        if target_column not in self._rawdf.columns:
            raise KeyError(f"Unknown target column: {target_column}")
        if target:
            return self._rawdf.loc[:, [target_column]].copy(deep=True)
        return self._rawdf.loc[:, self._rawdf.columns != target_column].copy(deep=True)

    def ordered_features(self) -> list[str]:
        self._ensure_frozen()
        return list(self._rawdf.columns)

    def __len__(self) -> int:
        self._ensure_frozen()
        return int(self._rawdf.shape[0])

    def __getitem__(self, key) -> pd.DataFrame:
        self._ensure_frozen()
        if isinstance(key, int):
            return self._rawdf.iloc[[key]].copy(deep=True)
        if isinstance(key, slice):
            return self._rawdf.iloc[key].copy(deep=True)
        if isinstance(key, str):
            if key not in self._rawdf.columns:
                raise KeyError(f"Unknown feature name: {key}")
            return self._rawdf.loc[:, [key]].copy(deep=True)
        raise TypeError(
            "DatasetObject only supports int/slice row indexing or str column indexing"
        )

    def clone(self) -> DatasetObject:
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__ = deepcopy(self.__dict__)
        clone._freeze = False
        return clone
