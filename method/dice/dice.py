from __future__ import annotations

import numpy as np
import pandas as pd
from dice_ml import Data as DiceData
from dice_ml import Dice as DiceExplainer
from dice_ml import Model as DiceModel

from dataset.dataset_object import DatasetObject
from method.dice.support import (
    BlackBoxModelTypes,
    RecourseModelAdapter,
    ensure_supported_target_model,
    resolve_feature_groups,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


class _DiceModelWrapper:
    def __init__(self, adapter: RecourseModelAdapter):
        self._adapter = adapter
        self.classes_ = adapter.classes_

    def predict_proba(self, X):
        return np.asarray(self._adapter.predict_proba(X), dtype=np.float64)

    def predict(self, X):
        prediction = self._adapter.predict(X)
        return np.asarray(prediction)


@register("dice")
class DiceMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        num: int = 1,
        posthoc_sparsity_param: float = 0.1,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, BlackBoxModelTypes, "DiceMethod")

        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class
        self._num = int(num)
        self._posthoc_sparsity_param = float(posthoc_sparsity_param)

        if self._num < 1:
            raise ValueError("num must be >= 1")
        if self._posthoc_sparsity_param < 0:
            raise ValueError("posthoc_sparsity_param must be >= 0")
        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for DiceMethod.fit()")

        with seed_context(self._seed):
            feature_groups = resolve_feature_groups(trainset)
            self._feature_names = list(feature_groups.feature_names)
            self._target_column = str(trainset.target_column)
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )
            train_df = pd.concat(
                [trainset.get(target=False), trainset.get(target=True)], axis=1
            )
            self._dice_data = DiceData(
                dataframe=train_df,
                continuous_features=list(feature_groups.continuous),
                outcome_name=self._target_column,
            )
            self._dice_model = DiceModel(
                model=_DiceModelWrapper(self._adapter),
                backend="sklearn",
            )
            self._dice = DiceExplainer(
                self._dice_data,
                self._dice_model,
                method="random",
            )
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        rows: list[pd.Series] = []
        desired_class = (
            "opposite" if self._desired_class is None else self._desired_class
        )

        with seed_context(self._seed):
            for _, row in factuals.iterrows():
                query = row.to_frame().T
                try:
                    explanation = self._dice.generate_counterfactuals(
                        query,
                        total_CFs=self._num,
                        desired_class=desired_class,
                        posthoc_sparsity_param=self._posthoc_sparsity_param,
                    )
                    example_list = getattr(explanation, "cf_examples_list", None) or []
                    if not example_list:
                        raise ValueError("DiCE returned no counterfactual examples")
                    counterfactual_df = example_list[0].final_cfs_df
                    if counterfactual_df is None or counterfactual_df.empty:
                        raise ValueError("DiCE returned an empty counterfactual frame")
                    feature_df = counterfactual_df.drop(
                        columns=[self._target_column], errors="ignore"
                    )
                    feature_df = feature_df.reindex(columns=self._feature_names)
                    rows.append(feature_df.iloc[0].copy(deep=True))
                except Exception:
                    rows.append(pd.Series(np.nan, index=self._feature_names))

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )
