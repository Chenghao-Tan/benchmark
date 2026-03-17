from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from preprocess.preprocess_object import PreProcessObject
from preprocess.preprocess_utils import (
    dataset_has_attr,
    ensure_flag_absent,
    resolve_feature_metadata,
)
from utils.registry import register
from utils.seed import seed_context


@register("encode")
class EncodePreProcess(PreProcessObject):
    @staticmethod
    def _resolve_encoding(encoding: str | None) -> str | None:
        if encoding is None:
            return None
        if not isinstance(encoding, str):
            raise TypeError("encoding must be str or None")

        encoding = encoding.lower()
        if encoding not in {"none", "onehot", "thermometer"}:
            raise ValueError(f"Unsupported encoding option: {encoding}")
        return encoding

    @staticmethod
    def _resolve_filter(filter: list[str] | None) -> list[str] | None:
        if filter is None:
            return None
        if not isinstance(filter, list):
            raise TypeError("EncodePreProcess filter must be a list[str] or None")

        resolved_filter: list[str] = []
        seen: set[str] = set()
        for feature_name in filter:
            if not isinstance(feature_name, str):
                raise TypeError(
                    "EncodePreProcess filter must contain str feature names only"
                )
            if feature_name in seen:
                raise ValueError(
                    f"EncodePreProcess filter contains duplicated feature: {feature_name}"
                )
            seen.add(feature_name)
            resolved_filter.append(feature_name)
        return resolved_filter

    @staticmethod
    def _resolve_encode_columns(
        df: pd.DataFrame,
        target_column: str,
        raw_feature_type: dict[str, str],
        filter: list[str] | None,
    ) -> list[str]:
        categorical_columns = []
        for column in df.columns:
            if column == target_column:
                continue
            if raw_feature_type.get(column, "").lower() == "categorical":
                categorical_columns.append(column)

        if filter is None:
            return categorical_columns
        if len(filter) == 0:
            return categorical_columns

        selected_columns: list[str] = []
        for feature_name in filter:
            if feature_name == target_column:
                raise ValueError(
                    "EncodePreProcess filter must not contain the target feature"
                )
            if feature_name not in df.columns:
                raise ValueError(
                    f"EncodePreProcess filter contains unknown feature: {feature_name}"
                )
            if raw_feature_type.get(feature_name, "").lower() != "categorical":
                raise ValueError(
                    f"EncodePreProcess filter contains non-categorical feature: {feature_name}"
                )
            selected_columns.append(feature_name)
        return selected_columns

    @staticmethod
    def _build_onehot(series: pd.Series, feature_name: str) -> pd.DataFrame:
        categories = list(pd.Index(series.dropna().unique()))
        categorical = pd.Categorical(series, categories=categories, ordered=True)
        encoded = pd.get_dummies(categorical, dtype="float64")
        encoded.index = series.index
        encoded.columns = [
            f"{feature_name}_cat_{index}" for index in range(1, encoded.shape[1] + 1)
        ]
        return encoded

    @staticmethod
    def _build_thermometer(series: pd.Series, feature_name: str) -> pd.DataFrame:
        categories = list(pd.Index(series.dropna().unique()))
        categorical = pd.Categorical(series, categories=categories, ordered=True)
        codes = pd.Series(categorical.codes, index=series.index)

        encoded_columns: dict[str, pd.Series] = {}
        for index in range(len(categories)):
            column_name = f"{feature_name}_cat_{index + 1}"
            encoded_columns[column_name] = (codes >= index).astype("float64")
        return pd.DataFrame(encoded_columns, index=series.index)

    @staticmethod
    def _build_encoded_metadata(
        dataset: DatasetObject,
        final_columns: list[str],
        encoded_sources: dict[str, str],
    ) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
        raw_feature_type = dataset.attr("raw_feature_type")
        raw_feature_mutability = dataset.attr("raw_feature_mutability")
        raw_feature_actionability = dataset.attr("raw_feature_actionability")

        encoded_feature_type: dict[str, str] = {}
        encoded_feature_mutability: dict[str, bool] = {}
        encoded_feature_actionability: dict[str, str] = {}

        for column in final_columns:
            if column in encoded_sources:
                source_feature = encoded_sources[column]
                encoded_feature_type[column] = "binary"
                encoded_feature_mutability[column] = bool(
                    raw_feature_mutability[source_feature]
                )
                encoded_feature_actionability[column] = str(
                    raw_feature_actionability[source_feature]
                )
            else:
                encoded_feature_type[column] = str(raw_feature_type[column])
                encoded_feature_mutability[column] = bool(
                    raw_feature_mutability[column]
                )
                encoded_feature_actionability[column] = str(
                    raw_feature_actionability[column]
                )

        return (
            encoded_feature_type,
            encoded_feature_mutability,
            encoded_feature_actionability,
        )

    def __init__(
        self,
        seed: int | None = None,
        encoding: str | None = "onehot",
        filter: list[str] | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._encoding: str | None = self._resolve_encoding(encoding)
        self._filter: list[str] | None = self._resolve_filter(filter)

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            ensure_flag_absent(input, "encoding")

            if self._encoding is None:
                return input
            if self._encoding == "none":
                return input

            df = input.snapshot()
            target_column = input.target_column

            raw_feature_type = input.attr("raw_feature_type")
            selected_columns = self._resolve_encode_columns(
                df=df,
                target_column=target_column,
                raw_feature_type=raw_feature_type,
                filter=self._filter,
            )

            final_parts: list[pd.DataFrame] = []
            encoded_sources: dict[str, str] = {}
            for column in df.columns:
                if column == target_column:
                    final_parts.append(df.loc[:, [column]].copy(deep=True))
                elif column not in selected_columns:
                    final_parts.append(df.loc[:, [column]].copy(deep=True))
                elif self._encoding == "onehot":
                    encoded = self._build_onehot(df[column], column)
                    final_parts.append(encoded)
                    for encoded_column in encoded.columns:
                        encoded_sources[encoded_column] = column
                elif self._encoding == "thermometer":
                    encoded = self._build_thermometer(df[column], column)
                    final_parts.append(encoded)
                    for encoded_column in encoded.columns:
                        encoded_sources[encoded_column] = column
                else:
                    raise ValueError(f"Unsupported encoding option: {self._encoding}")

            final_df = pd.concat(final_parts, axis=1)
            (
                encoded_feature_type,
                encoded_feature_mutability,
                encoded_feature_actionability,
            ) = self._build_encoded_metadata(
                dataset=input,
                final_columns=list(final_df.columns),
                encoded_sources=encoded_sources,
            )

            input.update("encoding", self._encoding, df=final_df)
            input.update("encoded_feature_type", encoded_feature_type)
            input.update("encoded_feature_mutability", encoded_feature_mutability)
            input.update("encoded_feature_actionability", encoded_feature_actionability)
            return input


@register("scale")
class ScalePreProcess(PreProcessObject):
    @staticmethod
    def _resolve_scaling(scaling: str | None) -> str | None:
        if scaling is None:
            return None
        if not isinstance(scaling, str):
            raise TypeError("scaling must be str or None")

        scaling = scaling.lower()
        if scaling not in {"none", "standardize", "normalize"}:
            raise ValueError(f"Unsupported scaling option: {scaling}")
        return scaling

    @staticmethod
    def _resolve_filter(filter: list[str] | None) -> list[str] | None:
        if filter is None:
            return None
        if not isinstance(filter, list):
            raise TypeError("ScalePreProcess filter must be a list[str] or None")

        resolved_filter: list[str] = []
        seen: set[str] = set()
        for feature_name in filter:
            if not isinstance(feature_name, str):
                raise TypeError(
                    "ScalePreProcess filter must contain str feature names only"
                )
            if feature_name in seen:
                raise ValueError(
                    f"ScalePreProcess filter contains duplicated feature: {feature_name}"
                )
            seen.add(feature_name)
            resolved_filter.append(feature_name)
        return resolved_filter

    @staticmethod
    def _resolve_scale_columns(
        df: pd.DataFrame,
        target_column: str,
        feature_type: dict[str, str],
        filter: list[str] | None,
    ) -> list[str]:
        numerical_columns = []
        for column in df.columns:
            if column == target_column:
                continue
            if feature_type.get(column, "").lower() == "numerical":
                numerical_columns.append(column)

        if filter is None:
            return numerical_columns
        if len(filter) == 0:
            return numerical_columns

        selected_columns: list[str] = []
        for feature_name in filter:
            if feature_name == target_column:
                raise ValueError(
                    "ScalePreProcess filter must not contain the target feature"
                )
            if feature_name not in df.columns:
                raise ValueError(
                    f"ScalePreProcess filter contains unknown feature: {feature_name}"
                )
            if feature_type.get(feature_name, "").lower() != "numerical":
                raise ValueError(
                    f"ScalePreProcess filter contains non-numerical feature: {feature_name}"
                )
            selected_columns.append(feature_name)
        return selected_columns

    @staticmethod
    def _compute_feature_ranges(
        df: pd.DataFrame, target_column: str, feature_type: dict[str, str]
    ) -> dict[str, float]:
        feature_ranges: dict[str, float] = {}
        for column in df.columns:
            if column == target_column:
                continue
            if feature_type.get(column, "").lower() == "numerical":
                feature_ranges[column] = float(df[column].max() - df[column].min())
        return feature_ranges

    def __init__(
        self,
        seed: int | None = None,
        scaling: str | None = "standardize",
        filter: list[str] | None = None,
        range: bool = True,
        **kwargs,
    ):
        self._seed = seed
        self._scaling: str | None = self._resolve_scaling(scaling)
        self._filter: list[str] | None = self._resolve_filter(filter)
        self._range: bool = range

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            ensure_flag_absent(input, "scaling")

            df = input.snapshot()
            target_column = input.target_column
            feature_type, _, _ = resolve_feature_metadata(input)

            if self._range:
                if dataset_has_attr(input, "range"):
                    pass
                else:
                    input.update(
                        "range",
                        self._compute_feature_ranges(df, target_column, feature_type),
                    )

            if self._scaling is None:
                return input
            if self._scaling == "none":
                return input

            selected_columns = self._resolve_scale_columns(
                df=df,
                target_column=target_column,
                feature_type=feature_type,
                filter=self._filter,
            )

            scaled_df = df.copy(deep=True)
            for column in selected_columns:
                series = scaled_df[column].astype("float64")

                if self._scaling == "standardize":
                    mean_value = float(series.mean())
                    std_value = float(series.std(ddof=0))
                    if std_value == 0.0:
                        scaled_df[column] = 0.0
                    else:
                        scaled_df[column] = (series - mean_value) / std_value
                elif self._scaling == "normalize":
                    min_value = float(series.min())
                    max_value = float(series.max())
                    scale_value = max_value - min_value
                    if scale_value == 0.0:
                        scaled_df[column] = 0.0
                    else:
                        scaled_df[column] = (series - min_value) / scale_value
                else:
                    raise ValueError(f"Unsupported scaling option: {self._scaling}")

            input.update("scaling", self._scaling, df=scaled_df)
            return input


@register("reorder")
class ReorderPreProcess(PreProcessObject):
    @staticmethod
    def _resolve_order(order: list[str] | None) -> list[str]:
        if order is None:
            raise ValueError("order must not be None")
        if not isinstance(order, list):
            raise TypeError("order must be a list[str]")
        if len(order) == 0:
            raise ValueError("order must not be empty")

        resolved_order: list[str] = []
        seen: set[str] = set()
        for feature_name in order:
            if not isinstance(feature_name, str):
                raise TypeError("order must contain str feature names only")
            if feature_name in seen:
                raise ValueError(f"order contains duplicated feature: {feature_name}")
            seen.add(feature_name)
            resolved_order.append(feature_name)
        return resolved_order

    def __init__(
        self,
        seed: int | None = None,
        order: list[str] | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._order: list[str] = self._resolve_order(order)

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            ensure_flag_absent(input, "reordered")

            df = input.snapshot()
            target_column = input.target_column

            feature_columns = [
                column for column in df.columns if column != target_column
            ]
            for feature_name in self._order:
                if feature_name == target_column:
                    raise ValueError(
                        "ReorderPreProcess order must not contain the target feature"
                    )
                if feature_name not in feature_columns:
                    raise ValueError(
                        f"ReorderPreProcess order contains unknown feature: {feature_name}"
                    )

            reordered_columns = list(self._order) + [target_column]
            reordered_df = df.loc[:, reordered_columns].copy(deep=True)

            input.update("reordered", list(self._order), df=reordered_df)
            return input


@register("split")
class SplitPreProcess(PreProcessObject):
    @staticmethod
    def _resolve_split(split: float | int) -> float | int:
        if isinstance(split, int):
            split = int(split)
            if split < 1:
                raise ValueError("integer split must be >= 1")
            return split
        if isinstance(split, float):
            split = float(split)
            if split <= 0 or split >= 1:
                raise ValueError("float split must satisfy 0 < split < 1")
            return split
        raise TypeError("split must be float or int")

    @staticmethod
    def _resolve_sample(sample: int | None) -> int | None:
        if sample is None:
            return None
        if isinstance(sample, int):
            sample = int(sample)
            if sample < 1:
                raise ValueError("sample must be >= 1 when provided")
            return sample
        raise TypeError("sample must be int or None")

    def __init__(
        self,
        seed: int | None = None,
        split: float | int = 0.2,
        sample: int | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._split: float | int = self._resolve_split(split)
        self._sample: int | None = self._resolve_sample(sample)

    def transform(self, input: DatasetObject) -> tuple[DatasetObject, DatasetObject]:
        with seed_context(self._seed):
            ensure_flag_absent(input, "trainset")
            ensure_flag_absent(input, "testset")

            df = input.snapshot()
            num_rows = int(df.shape[0])

            if isinstance(self._split, float):
                test_size = int(math.ceil(num_rows * self._split))
            elif isinstance(self._split, int):
                test_size = int(self._split)
            else:
                raise ValueError(f"Unsupported split option: {self._split}")

            if test_size < 1:
                raise ValueError("SplitPreProcess requires a test split size >= 1")
            if test_size >= num_rows:
                raise ValueError(
                    "SplitPreProcess requires at least one training sample"
                )

            shuffled_positions = np.random.permutation(num_rows)
            test_positions = shuffled_positions[:test_size]
            train_positions = shuffled_positions[test_size:]

            train_df = df.iloc[train_positions].copy(deep=True)
            test_df = df.iloc[test_positions].copy(deep=True)

            if self._sample is None:
                pass
            elif self._sample > test_df.shape[0]:
                raise ValueError("SplitPreProcess sample exceeds split testset size")
            else:
                sampled_positions = np.random.permutation(test_df.shape[0])[
                    : self._sample
                ]
                test_df = test_df.iloc[sampled_positions].copy(deep=True)

            testset = input.clone()
            trainset = input  # Reuse input

            trainset.update("trainset", True, df=train_df)
            testset.update("testset", True, df=test_df)

            return trainset, testset


@register("finalize")
class FinalizePreProcess(PreProcessObject):
    def __init__(self, seed: int | None = None, **kwargs):
        self._seed = seed

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            input.freeze()
            return input
