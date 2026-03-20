from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from preprocess.preprocess_utils import resolve_feature_metadata
from utils.registry import register


@register("cfrl_adult_metrics")
class CfrlAdultMetricsEvaluation(EvaluationObject):
    def __init__(self, target_model=None, **kwargs):
        self._target_model = target_model

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        (
            factual_features,
            counterfactual_features,
            evaluation_mask,
            success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)
        selected_mask = evaluation_mask & success_mask
        if int(selected_mask.sum()) == 0:
            return pd.DataFrame(
                [
                    {
                        "sparsity_cat_l0": float("nan"),
                        "sparsity_num_l1": float("nan"),
                        "immutability_violation_rate": float("nan"),
                    }
                ]
            )

        factual_selected = factual_features.loc[selected_mask.to_numpy()].copy(deep=True)
        counterfactual_selected = counterfactual_features.loc[
            selected_mask.to_numpy()
        ].copy(deep=True)

        feature_type, feature_mutability, feature_actionability = resolve_feature_metadata(
            factuals
        )
        categorical_columns = [
            column
            for column in factual_selected.columns
            if str(feature_type[column]).lower() != "numerical"
        ]
        numerical_columns = [
            column
            for column in factual_selected.columns
            if str(feature_type[column]).lower() == "numerical"
        ]
        immutable_columns = [
            column
            for column in factual_selected.columns
            if (not bool(feature_mutability[column]))
            or str(feature_actionability[column]).lower() in {"none", "same"}
        ]

        cat_l0 = float(
            (
                factual_selected.loc[:, categorical_columns]
                != counterfactual_selected.loc[:, categorical_columns]
            )
            .to_numpy(dtype=np.float32)
            .sum(axis=1)
            .mean()
        )
        num_l1 = float(
            np.abs(
                factual_selected.loc[:, numerical_columns].to_numpy(dtype=np.float32)
                - counterfactual_selected.loc[:, numerical_columns].to_numpy(dtype=np.float32)
            )
            .sum(axis=1)
            .mean()
        )
        if immutable_columns:
            immutable_violation_rate = float(
                (
                    factual_selected.loc[:, immutable_columns]
                    != counterfactual_selected.loc[:, immutable_columns]
                )
                .any(axis=1)
                .mean()
            )
        else:
            immutable_violation_rate = 0.0

        return pd.DataFrame(
            [
                {
                    "sparsity_cat_l0": cat_l0,
                    "sparsity_num_l1": num_l1,
                    "immutability_violation_rate": immutable_violation_rate,
                }
            ]
        )
