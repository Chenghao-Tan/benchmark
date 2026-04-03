"""Shared helpers for evaluation modules."""

from __future__ import annotations

import pandas as pd
import torch

from dataset.dataset_object import DatasetObject


def resolve_evaluation_inputs(
    factuals: DatasetObject, counterfactuals: DatasetObject
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Validate and align factual/counterfactual datasets for evaluation.

    The returned masks separate rows that should be considered by a metric from
    rows where the counterfactual itself is invalid because it contains missing
    values.

    Args:
        factuals: Frozen dataset with original examples.
        counterfactuals: Frozen dataset marked as counterfactual output.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Factual
        features, counterfactual features, evaluation mask, and success mask.

    Raises:
        ValueError: If dataset roles, target columns, or feature layouts do not
            match.
        TypeError: If ``evaluation_filter`` has an unsupported type.
    """
    if getattr(factuals, "counterfactual", False):
        raise ValueError("factuals must not be marked as counterfactual")
    if not getattr(counterfactuals, "counterfactual", False):
        raise ValueError("counterfactuals must be marked as counterfactual")
    if factuals.target_column != counterfactuals.target_column:
        raise ValueError(
            "factuals and counterfactuals must share the same target column"
        )

    factual_features = factuals.get(target=False)
    counterfactual_features = counterfactuals.get(target=False)
    if factual_features.shape != counterfactual_features.shape:
        raise ValueError(
            "factuals and counterfactuals must have the same feature shape"
        )
    if list(factual_features.columns) != list(counterfactual_features.columns):
        raise ValueError(
            "factuals and counterfactuals must have the same feature columns"
        )
    counterfactual_features = counterfactual_features.loc[factual_features.index]

    evaluation_mask = pd.Series(True, index=factual_features.index, dtype=bool)
    if not hasattr(counterfactuals, "evaluation_filter"):
        success_mask = ~counterfactual_features.isna().any(axis=1)
        return factual_features, counterfactual_features, evaluation_mask, success_mask

    raw_filter = counterfactuals.attr("evaluation_filter")
    if isinstance(raw_filter, pd.Series):
        evaluation_mask = raw_filter.astype(bool)
    elif isinstance(raw_filter, pd.DataFrame):
        if raw_filter.shape[1] == 0:
            raise ValueError("evaluation_filter must contain at least one column")
        if raw_filter.shape[1] == 1:
            evaluation_mask = raw_filter.iloc[:, 0].astype(bool)
        else:
            evaluation_mask = raw_filter.astype(bool).all(axis=1)
    else:
        raise TypeError("evaluation_filter must be a pandas Series or DataFrame")

    if evaluation_mask.shape[0] != counterfactual_features.shape[0]:
        raise ValueError("evaluation_filter length must match counterfactual rows")
    evaluation_mask = evaluation_mask.loc[factual_features.index]

    success_mask = ~counterfactual_features.isna().any(axis=1)
    return (
        factual_features,
        counterfactual_features,
        evaluation_mask,
        success_mask,
    )


def to_float_tensor(df: pd.DataFrame) -> torch.Tensor:
    """Convert a numeric DataFrame into a float tensor for metric computation.

    Args:
        df: Numeric feature matrix.

    Returns:
        torch.Tensor: Tensor with ``float32`` dtype.

    Raises:
        ValueError: If the DataFrame contains non-numeric values.
    """
    try:
        values = df.to_numpy(dtype="float32")
    except ValueError as error:
        raise ValueError("Evaluation requires numeric feature values") from error
    return torch.tensor(values, dtype=torch.float32)
