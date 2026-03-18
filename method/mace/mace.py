from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.mace.library.mace import generateExplanations
from method.mace.support import ensure_supported_target_model, validate_counterfactuals
from method.method_object import MethodObject
from model.model_object import ModelObject
from model.randomforest.randomforest import RandomForestModel
from utils.caching import get_cache_dir
from utils.registry import register
from utils.seed import seed_context


@dataclass
class _MaceAttribute:
    attr_name_kurz: str
    attr_type: str
    lower_bound: float
    upper_bound: float
    mutability: bool
    actionability: str


class _MaceDatasetWrapper:
    def __init__(
        self,
        feature_names: list[str],
        bounds: dict[str, tuple[float, float]],
        mutability: dict[str, bool],
        actionability: dict[str, str],
    ):
        self.dataset_name = "toydata"
        self.is_one_hot = False
        self._feature_names = list(feature_names)
        self._short_names = {
            feature: f"x{i}" for i, feature in enumerate(feature_names)
        }
        self._inverse_short_names = {
            short_name: feature_name
            for feature_name, short_name in self._short_names.items()
        }
        self.attributes_kurz: dict[str, _MaceAttribute] = {}

        for feature_name in feature_names:
            short_name = self._short_names[feature_name]
            lower_bound, upper_bound = bounds[feature_name]
            self.attributes_kurz[short_name] = _MaceAttribute(
                attr_name_kurz=short_name,
                attr_type="numeric-real",
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                mutability=bool(mutability[feature_name]),
                actionability=str(actionability[feature_name]).lower(),
            )
        self.attributes_kurz["y"] = _MaceAttribute(
            attr_name_kurz="y",
            attr_type="binary",
            lower_bound=0.0,
            upper_bound=1.0,
            mutability=False,
            actionability="none",
        )

    def getInputAttributeNames(self, kind: str = "kurz"):
        return [self._short_names[feature_name] for feature_name in self._feature_names]

    def getOutputAttributeNames(self, kind: str = "kurz"):
        return ["y"]

    def getInputOutputAttributeNames(self, kind: str = "kurz"):
        return self.getInputAttributeNames(kind) + self.getOutputAttributeNames(kind)

    def getMutableAttributeNames(self, kind: str = "kurz"):
        return [
            self._short_names[feature_name]
            for feature_name in self._feature_names
            if self.attributes_kurz[self._short_names[feature_name]].mutability
        ]

    def getOneHotAttributesNames(self, kind: str = "kurz"):
        return []

    def getNonHotAttributesNames(self, kind: str = "kurz"):
        return self.getInputAttributeNames(kind)

    def getSiblingsFor(self, attr_name_kurz: str):
        return [attr_name_kurz]

    def getDictOfSiblings(self, kind: str = "kurz"):
        return {"cat": {}, "ord": {}}

    def factual_to_short_dict(
        self,
        factual: pd.Series,
        predicted_label: int,
    ) -> dict[str, float | bool]:
        output = {
            self._short_names[feature_name]: float(factual[feature_name])
            for feature_name in self._feature_names
        }
        output["y"] = bool(predicted_label)
        return output

    def short_dict_to_feature_row(self, sample: dict[str, object]) -> pd.Series:
        row = {}
        for short_name, feature_name in self._inverse_short_names.items():
            value = sample.get(short_name, np.nan)
            row[feature_name] = float(value) if value is not None else np.nan
        return pd.Series(row, index=self._feature_names, dtype="float64")


@register("mace")
class MaceMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        norm_type: list[str] | None = None,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, (RandomForestModel,), "MaceMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class if desired_class is not None else 1
        self._norm_type = list(norm_type or ["zero_norm"])

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for MaceMethod.fit()")

        with seed_context(self._seed):
            self._feature_names = list(trainset.get(target=False).columns)
            feature_df = trainset.get(target=False)
            raw_feature_mutability = trainset.attr("raw_feature_mutability")
            raw_feature_actionability = trainset.attr("raw_feature_actionability")
            bounds = {
                feature_name: (
                    float(feature_df[feature_name].min()),
                    float(feature_df[feature_name].max()),
                )
                for feature_name in self._feature_names
            }
            mutability = {
                feature_name: bool(raw_feature_mutability[feature_name])
                for feature_name in self._feature_names
            }
            actionability = {
                feature_name: str(raw_feature_actionability[feature_name]).lower()
                for feature_name in self._feature_names
            }
            self._dataset_wrapper = _MaceDatasetWrapper(
                self._feature_names,
                bounds,
                mutability,
                actionability,
            )
            self._explanation_dir = Path(get_cache_dir("mace")) / "__explanation_log"
            self._explanation_dir.mkdir(parents=True, exist_ok=True)
            self._is_trained = True

    def _predict_label(self, factuals: pd.DataFrame) -> np.ndarray:
        prediction = self._target_model.get_prediction(factuals, proba=False)
        return prediction.detach().cpu().numpy().argmax(axis=1)

    def get_counterfactuals(self, factuals: pd.DataFrame):
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        predicted_labels = self._predict_label(factuals)
        rows = []

        with seed_context(self._seed):
            for row_index, (_, row) in enumerate(factuals.iterrows()):
                factual_sample = self._dataset_wrapper.factual_to_short_dict(
                    row, int(predicted_labels[row_index])
                )
                explanation_file_name = str(
                    self._explanation_dir / f"sample_{factuals.index[row_index]}.txt"
                )
                found_row = None
                for norm_type in self._norm_type:
                    result = generateExplanations(
                        "MACE_eps_1e-5",
                        explanation_file_name,
                        self._target_model._model,
                        self._dataset_wrapper,
                        factual_sample,
                        norm_type,
                    )
                    cfe_sample = (
                        result.get("cfe_sample") if isinstance(result, dict) else None
                    )
                    if cfe_sample:
                        found_row = self._dataset_wrapper.short_dict_to_feature_row(
                            cfe_sample
                        )
                        break
                if found_row is None:
                    found_row = pd.Series(
                        np.nan, index=self._feature_names, dtype="float64"
                    )
                rows.append(found_row)

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
