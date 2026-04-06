from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.cemsp.support import (
    BlackBoxModelTypes,
    CFSolver,
    FeatureGroups,
    MapSolver,
    RecourseModelAdapter,
    ensure_supported_target_model,
    find_cf,
    parse_feature_vector,
    resolve_feature_groups,
    resolve_target_classes,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("cemsp")
class CemspMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        replacement_policy: str = "positive_median",
        replacement_quantile: float = 0.5,
        replacement_statistic: str = "median",
        explicit_to_replace: dict[str, object] | list[object] | None = None,
        explicit_lower_bounds: dict[str, object] | list[object] | None = None,
        explicit_upper_bounds: dict[str, object] | list[object] | None = None,
        max_counterfactuals: int = 1,
        return_mode: str = "first",
        **kwargs,
    ):
        ensure_supported_target_model(target_model, BlackBoxModelTypes, "CemspMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._replacement_policy = str(replacement_policy).lower()
        self._replacement_quantile = float(replacement_quantile)
        self._replacement_statistic = str(replacement_statistic).lower()
        self._explicit_to_replace = explicit_to_replace
        self._explicit_lower_bounds = explicit_lower_bounds
        self._explicit_upper_bounds = explicit_upper_bounds
        self._max_counterfactuals = int(max_counterfactuals)
        self._return_mode = str(return_mode).lower()

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._max_counterfactuals < 1:
            raise ValueError("max_counterfactuals must be >= 1")
        if not (0.0 <= self._replacement_quantile <= 1.0):
            raise ValueError("replacement_quantile must be between 0 and 1")
        if self._replacement_statistic not in {"median", "mean", "quantile"}:
            raise ValueError(
                "replacement_statistic must be one of ['median', 'mean', 'quantile']"
            )
        if self._return_mode not in {"first", "best_l1"}:
            raise ValueError("return_mode must be either 'first' or 'best_l1'")
        if self._replacement_policy not in {
            "positive_median",
            "positive_mean",
            "positive_quantile",
            "explicit",
            "bounds",
        }:
            raise ValueError("Unsupported replacement_policy for CemspMethod")
        if self._explicit_to_replace is not None and (
            self._explicit_lower_bounds is not None or self._explicit_upper_bounds is not None
        ):
            raise ValueError(
                "explicit_to_replace is incompatible with explicit_lower_bounds/explicit_upper_bounds"
            )
        if self._replacement_policy == "explicit" and self._explicit_to_replace is None:
            raise ValueError(
                "replacement_policy='explicit' requires explicit_to_replace"
            )
        if self._replacement_policy == "bounds" and (
            self._explicit_lower_bounds is None and self._explicit_upper_bounds is None
        ):
            raise ValueError(
                "replacement_policy='bounds' requires explicit_lower_bounds and/or explicit_upper_bounds"
            )

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for CemspMethod.fit()")

        with seed_context(self._seed):
            feature_df = trainset.get(target=False)
            try:
                train_array = feature_df.to_numpy(dtype=np.float64)
            except ValueError as error:
                raise ValueError(
                    "CemspMethod requires fully numeric input features"
                ) from error
            if np.isnan(train_array).any():
                raise ValueError("CemspMethod does not support NaN values in trainset")

            self._feature_names = list(feature_df.columns)
            self._feature_groups = resolve_feature_groups(trainset)
            self._adapter = RecourseModelAdapter(self._target_model, self._feature_names)
            self._feature_intervals = {}
            for feature_name in self._feature_names:
                feature_values = feature_df[feature_name].to_numpy(dtype=np.float64)
                self._feature_intervals[feature_name] = np.unique(feature_values)

            self._predicted_train_classes = self._adapter.predict(feature_df)
            self._train_features = feature_df.copy(deep=True)
            self._prototype_by_class = {}
            for class_label in pd.Index(self._predicted_train_classes).unique().tolist():
                pool = self._train_features.loc[
                    self._predicted_train_classes == class_label
                ].copy(deep=True)
                if pool.empty:
                    continue
                self._prototype_by_class[class_label] = self._build_prototype(pool)

            if self._explicit_to_replace is not None:
                self._explicit_to_replace_vector = parse_feature_vector(
                    self._explicit_to_replace,
                    self._feature_names,
                    "explicit_to_replace",
                )
            else:
                self._explicit_to_replace_vector = None
            if self._explicit_lower_bounds is not None:
                self._explicit_lower_bounds_vector = parse_feature_vector(
                    self._explicit_lower_bounds,
                    self._feature_names,
                    "explicit_lower_bounds",
                )
            else:
                self._explicit_lower_bounds_vector = None
            if self._explicit_upper_bounds is not None:
                self._explicit_upper_bounds_vector = parse_feature_vector(
                    self._explicit_upper_bounds,
                    self._feature_names,
                    "explicit_upper_bounds",
                )
            else:
                self._explicit_upper_bounds_vector = None

            self._is_trained = True

    def _build_prototype(self, pool: pd.DataFrame) -> np.ndarray:
        values = []
        for feature_name in self._feature_names:
            feature_kind = self._feature_groups.feature_type[feature_name].lower()
            column = pool[feature_name]
            if feature_kind == "numerical":
                if self._replacement_statistic == "median":
                    value = float(column.median())
                elif self._replacement_statistic == "mean":
                    value = float(column.mean())
                else:
                    value = float(column.quantile(self._replacement_quantile))
            else:
                mode = column.mode(dropna=True)
                if mode.empty:
                    value = float(column.iloc[0])
                else:
                    value = float(mode.iloc[0])
            values.append(value)
        return np.asarray(values, dtype=np.float64)

    def _apply_constraints(self, factual: np.ndarray, replacement: np.ndarray) -> np.ndarray:
        adjusted = replacement.copy()
        for index, constraint in enumerate(self._feature_groups.plausibility_constraints):
            if constraint == "=":
                adjusted[index] = factual[index]
            elif constraint == ">=" and adjusted[index] < factual[index]:
                adjusted[index] = factual[index]
            elif constraint == "<=" and adjusted[index] > factual[index]:
                adjusted[index] = factual[index]
        for index, feature_name in enumerate(self._feature_names):
            feature_kind = self._feature_groups.feature_type[feature_name].lower()
            if feature_kind != "numerical":
                candidates = self._feature_intervals[feature_name]
                distances = np.abs(candidates - adjusted[index])
                adjusted[index] = float(candidates[int(np.argmin(distances))])
        return adjusted

    def _build_replacement_vector(
        self,
        factual: np.ndarray,
        desired_class: int | str,
    ) -> np.ndarray:
        if self._explicit_to_replace_vector is not None:
            replacement = self._explicit_to_replace_vector.copy()
        elif (
            self._explicit_lower_bounds_vector is not None
            or self._explicit_upper_bounds_vector is not None
        ):
            replacement = factual.copy()
            if self._explicit_lower_bounds_vector is not None:
                replacement = np.maximum(replacement, self._explicit_lower_bounds_vector)
            if self._explicit_upper_bounds_vector is not None:
                replacement = np.minimum(replacement, self._explicit_upper_bounds_vector)
        else:
            if desired_class not in self._prototype_by_class:
                raise RuntimeError(
                    f"No replacement prototype could be built for desired class {desired_class}"
                )
            replacement = self._prototype_by_class[desired_class].copy()
        return self._apply_constraints(factual, replacement)

    def _enumerate_counterfactuals(
        self,
        factual: np.ndarray,
        desired_class: int | str,
        to_replace: np.ndarray,
    ) -> list[np.ndarray]:
        if np.allclose(factual, to_replace, atol=1e-8, rtol=1e-8):
            return []

        mapsolver = MapSolver(len(self._feature_names))
        cfsolver = CFSolver(
            len(self._feature_names),
            self._adapter,
            factual,
            to_replace,
            desired_class,
        )

        counterfactuals: list[np.ndarray] = []
        for _, cf, _ in find_cf(cfsolver, mapsolver):
            counterfactuals.append(np.asarray(cf, dtype=np.float64).reshape(-1))
            if len(counterfactuals) >= self._max_counterfactuals:
                break
        return counterfactuals

    def _select_counterfactual(
        self, factual: np.ndarray, counterfactuals: list[np.ndarray]
    ) -> np.ndarray | None:
        if not counterfactuals:
            return None
        if self._return_mode == "first":
            return counterfactuals[0]

        best_index = int(
            np.argmin(
                [float(np.abs(counterfactual - factual).sum()) for counterfactual in counterfactuals]
            )
        )
        return counterfactuals[best_index]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        with seed_context(self._seed):
            original_predictions = self._adapter.predict_label_indices(factuals)
            desired_classes = resolve_target_classes(
                self._target_model,
                original_predictions,
                self._desired_class,
            )

            rows: list[np.ndarray] = []
            for row_index, (_, row) in enumerate(factuals.iterrows()):
                factual = row.to_numpy(dtype=np.float64)
                desired_class = desired_classes[row_index]
                to_replace = self._build_replacement_vector(factual, desired_class)
                counterfactuals_list = self._enumerate_counterfactuals(
                    factual,
                    desired_class,
                    to_replace,
                )
                candidate = self._select_counterfactual(factual, counterfactuals_list)
                if candidate is None:
                    rows.append(np.full(len(self._feature_names), np.nan, dtype=np.float64))
                else:
                    rows.append(candidate)

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )
