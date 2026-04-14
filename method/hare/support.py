from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from model.randomforest.randomforest import RandomForestModel
from preprocess.preprocess_utils import dataset_has_attr, resolve_feature_metadata

TorchModelTypes = (LinearModel, MlpModel)
BlackBoxModelTypes = (LinearModel, MlpModel, RandomForestModel)


def ensure_supported_target_model(
    target_model: ModelObject,
    supported_types: Sequence[type[ModelObject]],
    method_name: str,
) -> None:
    if isinstance(target_model, tuple(supported_types)):
        return

    supported_names = ", ".join(cls.__name__ for cls in supported_types)
    raise TypeError(
        f"{method_name} supports target models [{supported_names}] only, "
        f"received {target_model.__class__.__name__}"
    )


@dataclass
class FeatureSchema:
    feature_names: list[str]
    feature_type: dict[str, str]
    feature_mutability: dict[str, bool]
    feature_actionability: dict[str, str]
    continuous_indices: list[int]
    binary_indices: list[int]
    categorical_groups: list[list[int]]
    immutable_indices: list[int]
    mutable_indices: list[int]
    mutable_continuous_indices: list[int]
    mutable_binary_indices: list[int]
    standalone_binary_indices: list[int]
    normalized_indices: list[int]


def validate_numeric_frame(df: pd.DataFrame, method_name: str) -> None:
    try:
        df.to_numpy(dtype="float32")
    except ValueError as error:
        raise ValueError(f"{method_name} requires fully numeric input features") from error


def resolve_feature_schema(dataset: DatasetObject) -> FeatureSchema:
    feature_df = dataset.get(target=False)
    feature_names = list(feature_df.columns)
    feature_type, feature_mutability, feature_actionability = resolve_feature_metadata(
        dataset
    )

    raw_feature_type = dataset.attr("raw_feature_type")
    encoding = dataset.attr("encoding") if dataset_has_attr(dataset, "encoding") else {}
    scaling = dataset.attr("scaling") if dataset_has_attr(dataset, "scaling") else {}

    categorical_groups: list[list[int]] = []
    for source_feature, encoded_columns in encoding.items():
        if str(raw_feature_type.get(source_feature, "")).lower() != "categorical":
            continue
        group = [feature_names.index(column) for column in encoded_columns if column in feature_names]
        if len(group) > 1:
            categorical_groups.append(group)

    grouped_binary_indices = {index for group in categorical_groups for index in group}
    continuous_indices: list[int] = []
    binary_indices: list[int] = []
    immutable_indices: list[int] = []
    mutable_indices: list[int] = []
    mutable_continuous_indices: list[int] = []
    mutable_binary_indices: list[int] = []
    normalized_indices: list[int] = []

    for index, feature_name in enumerate(feature_names):
        kind = str(feature_type[feature_name]).lower()
        actionability = str(feature_actionability[feature_name]).lower()
        is_mutable = bool(feature_mutability[feature_name]) and actionability not in {
            "none",
            "same",
        }

        if kind == "numerical":
            continuous_indices.append(index)
        if kind == "binary" or index in grouped_binary_indices:
            binary_indices.append(index)

        if is_mutable:
            mutable_indices.append(index)
            if index in continuous_indices:
                mutable_continuous_indices.append(index)
            if index in binary_indices:
                mutable_binary_indices.append(index)
        else:
            immutable_indices.append(index)

        if str(scaling.get(feature_name, "")).lower() == "normalize":
            normalized_indices.append(index)

    standalone_binary_indices = [
        index for index in binary_indices if index not in grouped_binary_indices
    ]

    return FeatureSchema(
        feature_names=feature_names,
        feature_type={key: str(value) for key, value in feature_type.items()},
        feature_mutability={key: bool(value) for key, value in feature_mutability.items()},
        feature_actionability={
            key: str(value).lower() for key, value in feature_actionability.items()
        },
        continuous_indices=continuous_indices,
        binary_indices=binary_indices,
        categorical_groups=categorical_groups,
        immutable_indices=immutable_indices,
        mutable_indices=mutable_indices,
        mutable_continuous_indices=mutable_continuous_indices,
        mutable_binary_indices=mutable_binary_indices,
        standalone_binary_indices=standalone_binary_indices,
        normalized_indices=normalized_indices,
    )


def to_feature_dataframe(
    values: pd.DataFrame | np.ndarray | torch.Tensor,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.loc[:, list(feature_names)].copy(deep=True)

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    if array.ndim == 1:
        array = array.reshape(1, -1)
    return pd.DataFrame(array, columns=list(feature_names))


def differentiable_predict_proba(
    target_model: ModelObject,
    X: torch.Tensor,
) -> torch.Tensor:
    ensure_supported_target_model(
        target_model,
        TorchModelTypes,
        "differentiable_predict_proba",
    )
    model = getattr(target_model, "_model", None)
    if model is None:
        raise RuntimeError("Target model has not been initialized")

    logits = model(X.to(target_model._device))
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    output_activation = str(
        getattr(target_model, "_output_activation_name", "softmax")
    ).lower()
    if output_activation == "sigmoid":
        positive_probability = torch.sigmoid(logits)
        return torch.cat([1.0 - positive_probability, positive_probability], dim=1)
    return torch.softmax(logits, dim=1)


class ModelAdapter:
    def __init__(self, target_model: ModelObject, feature_names: Sequence[str]):
        self._target_model = target_model
        self._feature_names = list(feature_names)
        self._class_to_index = target_model.get_class_to_index()
        self.supports_gradients = isinstance(target_model, TorchModelTypes)

    def get_ordered_features(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> pd.DataFrame:
        return to_feature_dataframe(X, self._feature_names)

    def predict_proba(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if isinstance(X, torch.Tensor) and self.supports_gradients:
            return differentiable_predict_proba(self._target_model, X)

        features = self.get_ordered_features(X)
        prediction = self._target_model.get_prediction(features, proba=True)
        if isinstance(prediction, torch.Tensor):
            return prediction.detach().cpu().numpy()
        return np.asarray(prediction, dtype=np.float32)

    def predict_label_indices(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        probabilities = self.predict_proba(X)
        if isinstance(probabilities, torch.Tensor):
            return probabilities.argmax(dim=1).detach().cpu().numpy()
        return np.asarray(probabilities).argmax(axis=1)


def resolve_target_indices(
    target_model: ModelObject,
    original_prediction: np.ndarray,
    desired_class: int | str | None,
) -> np.ndarray:
    class_to_index = target_model.get_class_to_index()
    if desired_class is not None:
        if desired_class not in class_to_index:
            raise ValueError("desired_class is invalid for the trained target model")
        return np.full(
            shape=original_prediction.shape,
            fill_value=int(class_to_index[desired_class]),
            dtype=np.int64,
        )

    if len(class_to_index) != 2:
        raise ValueError("desired_class=None is supported for binary classification only")
    return 1 - original_prediction.astype(np.int64, copy=False)


def _project_array_2d(
    values: np.ndarray,
    factual: np.ndarray,
    schema: FeatureSchema,
) -> np.ndarray:
    projected = np.asarray(values, dtype=np.float32).copy()
    if projected.ndim == 1:
        projected = projected.reshape(1, -1)

    factual_row = np.asarray(factual, dtype=np.float32).reshape(1, -1)
    grouped_indices = {index for group in schema.categorical_groups for index in group}
    immutable_indices = set(schema.immutable_indices)
    standalone_binary_indices = set(schema.standalone_binary_indices)
    normalized_indices = set(schema.normalized_indices)

    for group in schema.categorical_groups:
        group_array = np.array(group, dtype=np.int64)
        if group_array[0] in immutable_indices:
            projected[:, group_array] = factual_row[:, group_array]
            continue

        winner = projected[:, group_array].argmax(axis=1)
        projected[:, group_array] = 0.0
        projected[np.arange(projected.shape[0]), group_array[winner]] = 1.0

    for index, feature_name in enumerate(schema.feature_names):
        if index in grouped_indices:
            continue

        actionability = schema.feature_actionability[feature_name]
        if index in immutable_indices or actionability in {"none", "same"}:
            projected[:, index] = factual_row[:, index]
        elif actionability in {"same-or-increase", "increase"}:
            projected[:, index] = np.maximum(projected[:, index], factual_row[:, index])
        elif actionability in {"same-or-decrease", "decrease"}:
            projected[:, index] = np.minimum(projected[:, index], factual_row[:, index])

        if index in standalone_binary_indices:
            projected[:, index] = np.clip(np.round(projected[:, index]), 0.0, 1.0)
        elif index in normalized_indices:
            projected[:, index] = np.clip(projected[:, index], 0.0, 1.0)

    return projected


def project_actionability_array(
    values: np.ndarray,
    factual: np.ndarray,
    schema: FeatureSchema,
) -> np.ndarray:
    projected = _project_array_2d(values, factual, schema)
    if np.asarray(values).ndim == 1:
        return projected[0]
    return projected


def project_actionability_tensor_(
    values: torch.Tensor,
    factual: torch.Tensor,
    schema: FeatureSchema,
) -> torch.Tensor:
    with torch.no_grad():
        squeeze_output = False
        tensor = values
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
            squeeze_output = True

        factual_tensor = factual
        if factual_tensor.ndim == 1:
            factual_tensor = factual_tensor.unsqueeze(0)

        grouped_indices = {index for group in schema.categorical_groups for index in group}
        immutable_indices = set(schema.immutable_indices)
        standalone_binary_indices = set(schema.standalone_binary_indices)
        normalized_indices = set(schema.normalized_indices)

        for group in schema.categorical_groups:
            if group[0] in immutable_indices:
                tensor[:, group] = factual_tensor[:, group]
                continue

            winner = tensor[:, group].argmax(dim=1)
            tensor[:, group] = 0.0
            tensor[torch.arange(tensor.shape[0], device=tensor.device), [group[idx] for idx in winner.tolist()]] = 1.0

        for index, feature_name in enumerate(schema.feature_names):
            if index in grouped_indices:
                continue

            actionability = schema.feature_actionability[feature_name]
            if index in immutable_indices or actionability in {"none", "same"}:
                tensor[:, index] = factual_tensor[:, index]
            elif actionability in {"same-or-increase", "increase"}:
                tensor[:, index] = torch.maximum(tensor[:, index], factual_tensor[:, index])
            elif actionability in {"same-or-decrease", "decrease"}:
                tensor[:, index] = torch.minimum(tensor[:, index], factual_tensor[:, index])

            if index in standalone_binary_indices:
                tensor[:, index] = torch.clamp(torch.round(tensor[:, index]), 0.0, 1.0)
            elif index in normalized_indices:
                tensor[:, index] = torch.clamp(tensor[:, index], 0.0, 1.0)

        if squeeze_output:
            values.copy_(tensor.squeeze(0))
        else:
            values.copy_(tensor)
    return values


def actionable_mask(
    candidates: np.ndarray,
    factual: np.ndarray,
    schema: FeatureSchema,
    atol: float = 1e-6,
) -> np.ndarray:
    array = np.asarray(candidates, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    projected = project_actionability_array(array, factual, schema)
    return np.isclose(array, projected, atol=atol, rtol=0.0).all(axis=1)


def validate_counterfactuals(
    target_model: ModelObject,
    factuals: pd.DataFrame,
    candidates: pd.DataFrame,
    desired_class: int | str | None = None,
) -> pd.DataFrame:
    if list(candidates.columns) != list(factuals.columns):
        candidates = candidates.reindex(columns=factuals.columns)
    candidates = candidates.copy(deep=True)

    if candidates.shape[0] != factuals.shape[0]:
        raise ValueError("Candidates must preserve the number of factual rows")

    valid_rows = ~candidates.isna().any(axis=1)
    if not bool(valid_rows.any()):
        return candidates

    adapter = ModelAdapter(target_model, factuals.columns)
    original_prediction = adapter.predict_label_indices(factuals)
    target_prediction = resolve_target_indices(
        target_model=target_model,
        original_prediction=original_prediction,
        desired_class=desired_class,
    )

    candidate_prediction = adapter.predict_label_indices(candidates.loc[valid_rows])
    success_mask = pd.Series(False, index=candidates.index, dtype=bool)
    success_mask.loc[valid_rows] = (
        candidate_prediction.astype(np.int64, copy=False)
        == target_prediction[valid_rows.to_numpy()]
    )
    candidates.loc[~success_mask, :] = np.nan
    return candidates
