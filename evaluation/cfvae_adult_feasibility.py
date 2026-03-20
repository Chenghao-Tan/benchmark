from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from utils.registry import register


@register("cfvae_adult_feasibility")
class CfvaeAdultFeasibilityEvaluation(EvaluationObject):
    _EDUCATION_SCORE = {
        "School": 0,
        "HS-grad": 1,
        "Some-college": 2,
        "Assoc": 3,
        "Bachelors": 4,
        "Masters": 5,
        "Prof-school": 6,
        "Doctorate": 7,
    }

    def __init__(self, target_model=None, **kwargs):
        self._target_model = target_model

    @staticmethod
    def _decode_feature_frame(dataset: DatasetObject, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(dataset, "encoding"):
            return frame.copy(deep=True)

        encoding = dataset.attr("encoding")
        target_column = dataset.target_column
        raw_snapshot = dataset.attr("cfvae_raw_features")
        decoded_columns: dict[str, pd.Series] = {}

        for feature_name in raw_snapshot.columns:
            if feature_name == target_column:
                continue
            if feature_name not in encoding:
                if feature_name in frame.columns:
                    decoded_columns[feature_name] = frame[feature_name]
                continue

            encoded_columns = encoding[feature_name]
            if len(encoded_columns) == 1 and encoded_columns[0] == feature_name:
                decoded_columns[feature_name] = frame[feature_name]
                continue

            encoded_values = frame.loc[:, encoded_columns].to_numpy(dtype=np.float32)
            max_positions = encoded_values.argmax(axis=1)
            decoded_values = []
            for position in max_positions:
                column_name = encoded_columns[int(position)]
                if "_cat_" in column_name:
                    decoded_values.append(column_name.split("_cat_", maxsplit=1)[1])
                elif "_therm_" in column_name:
                    decoded_values.append(column_name.split("_therm_", maxsplit=1)[1])
                else:
                    decoded_values.append(column_name)
            decoded_columns[feature_name] = pd.Series(
                decoded_values,
                index=frame.index,
            )

        decoded_df = pd.DataFrame(decoded_columns, index=frame.index)
        numeric_bounds = dataset.attr("cfvae_numeric_bounds")
        scaling = getattr(dataset, "scaling", {})
        for feature_name, bounds in numeric_bounds.items():
            if feature_name not in decoded_df.columns:
                continue
            if scaling.get(feature_name) == "normalize":
                min_value = float(bounds["min"])
                max_value = float(bounds["max"])
                decoded_df[feature_name] = (
                    decoded_df[feature_name].astype(float) * (max_value - min_value)
                    + min_value
                )
        return decoded_df

    @staticmethod
    def _collapse_education(value: object) -> str:
        text = str(value)
        if text.startswith("Assoc"):
            return "Assoc"
        return text

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
                        "age_non_decrease_rate": float("nan"),
                        "age_education_feasibility_rate": float("nan"),
                        "cat_proximity": float("nan"),
                        "cont_proximity": float("nan"),
                    }
                ]
            )

        raw_snapshot = factuals.attr("cfvae_raw_features")
        factual_raw = raw_snapshot.loc[factual_features.index].loc[
            selected_mask.to_numpy()
        ].copy(deep=True)
        counterfactual_raw = self._decode_feature_frame(
            counterfactuals,
            counterfactual_features.loc[selected_mask.to_numpy()].copy(deep=True),
        )

        age_rate = float((counterfactual_raw["age"] >= factual_raw["age"]).mean())

        factual_education = factual_raw["education"].map(self._collapse_education)
        counterfactual_education = counterfactual_raw["education"].map(
            self._collapse_education
        )
        factual_score = factual_education.map(self._EDUCATION_SCORE).fillna(-1)
        counterfactual_score = counterfactual_education.map(self._EDUCATION_SCORE).fillna(
            -1
        )
        age_delta = counterfactual_raw["age"].astype(float) - factual_raw["age"].astype(
            float
        )
        education_feasible = (counterfactual_score >= factual_score) & (
            (counterfactual_score == factual_score) & (age_delta >= 0)
            | (counterfactual_score > factual_score) & (age_delta > 0)
        )
        age_education_rate = float(education_feasible.mean())

        categorical_columns = [
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "race",
            "gender",
        ]
        cat_proximity = float(
            -1.0
            * sum(
                (factual_raw[column].astype(str) != counterfactual_raw[column].astype(str)).sum()
                for column in categorical_columns
            )
            / factual_raw.shape[0]
        )

        continuous_columns = ["age", "hours_per_week"]
        mad_weights = factuals.attr("cfvae_numeric_mad")
        cont_distance = 0.0
        for column in continuous_columns:
            if column not in counterfactual_raw.columns:
                continue
            cont_distance += float(
                np.abs(
                    factual_raw[column].astype(float).to_numpy()
                    - counterfactual_raw[column].astype(float).to_numpy()
                ).sum()
                / mad_weights[column]
            )
        cont_proximity = float(-1.0 * cont_distance / factual_raw.shape[0])

        return pd.DataFrame(
            [
                {
                    "age_non_decrease_rate": age_rate,
                    "age_education_feasibility_rate": age_education_rate,
                    "cat_proximity": cat_proximity,
                    "cont_proximity": cont_proximity,
                }
            ]
        )
