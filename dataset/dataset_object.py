"""Base dataset abstraction used throughout the benchmark pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml


class DatasetObject(ABC):
    """Store dataset rows together with feature metadata used by the benchmark.

    A dataset starts in a mutable state so preprocessors can update the backing
    DataFrame and attach derived attributes. Once preprocessing is complete,
    call :meth:`freeze` to switch into read-only access mode for downstream
    training, generation, and evaluation code.
    """

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
        """Return a deep copy of the current mutable dataset contents.

        Returns:
            pd.DataFrame: Copy of the underlying dataset table.

        Raises:
            RuntimeError: If the dataset has already been frozen.
        """
        self._ensure_mutable()
        return self._rawdf.copy(deep=True)

    def update(self, flag: str, value: object, df: pd.DataFrame | None = None) -> bool:
        """Update mutable dataset state or attach derived metadata.

        Args:
            flag: Attribute name to set on the dataset instance.
            value: Value to store under ``flag``.
            df: Optional replacement DataFrame for the dataset rows.

        Returns:
            bool: ``True`` when the update completes.

        Raises:
            RuntimeError: If the dataset has already been frozen.
            ValueError: If ``flag`` is ``None``.
        """
        self._ensure_mutable()
        if flag is None:
            raise ValueError("flag must not be None")
        if df is not None:
            self._rawdf = df.copy(deep=True)
        setattr(self, flag, deepcopy(value))
        return True

    def attr(self, flag: str) -> object:
        """Return a deep copy of a public dataset attribute.

        Args:
            flag: Name of the attribute to retrieve.

        Returns:
            object: Copy of the requested attribute value.

        Raises:
            AttributeError: If the attribute is protected or missing.
        """
        if flag.startswith("_"):
            raise AttributeError(f"Access to protected member '{flag}' is forbidden")
        if not hasattr(self, flag):
            raise AttributeError(f"Unknown dataset attribute: {flag}")
        return deepcopy(getattr(self, flag))

    def freeze(self):
        """Mark the dataset as finalized and enable read-only accessors."""
        self._freeze = True

    def get(self, target: bool = False) -> pd.DataFrame:
        """Return either the feature matrix or the target column.

        Args:
            target: When ``True``, return only the target column. Otherwise,
                return all non-target feature columns.

        Returns:
            pd.DataFrame: Deep copy of the selected columns.

        Raises:
            RuntimeError: If the dataset is still mutable.
            KeyError: If the configured target column is missing.
        """
        self._ensure_frozen()
        target_column = self.target_column
        if target_column not in self._rawdf.columns:
            raise KeyError(f"Unknown target column: {target_column}")
        if target:
            return self._rawdf.loc[:, [target_column]].copy(deep=True)
        return self._rawdf.loc[:, self._rawdf.columns != target_column].copy(deep=True)

    def ordered_features(self) -> list[str]:
        """Return the frozen column order used by the dataset."""
        self._ensure_frozen()
        return list(self._rawdf.columns)

    def __len__(self) -> int:
        """Return the number of rows in the frozen dataset."""
        self._ensure_frozen()
        return int(self._rawdf.shape[0])

    def __getitem__(self, key) -> pd.DataFrame:
        """Access frozen rows or columns by index, slice, or column name.

        Args:
            key: Integer row index, row slice, or string column name.

        Returns:
            pd.DataFrame: Deep copy of the selected rows or column.

        Raises:
            RuntimeError: If the dataset is still mutable.
            KeyError: If a requested column name does not exist.
            TypeError: If ``key`` uses an unsupported type.
        """
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
        """Create a mutable copy of the dataset and its attached metadata.

        Returns:
            DatasetObject: Deep-copied dataset instance with ``_freeze`` reset.
        """
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__ = deepcopy(self.__dict__)
        clone._freeze = False
        return clone
