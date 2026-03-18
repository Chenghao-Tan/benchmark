from __future__ import annotations

import pandas as pd

from dataset.dataset_object import DatasetObject
from method.face.graph import graph_search
from method.face.support import (
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


@register("face")
class FaceMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        mode: str = "knn",
        fraction: float = 0.05,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, BlackBoxModelTypes, "FaceMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class
        self._mode = str(mode).lower()
        self._fraction = float(fraction)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._mode not in {"knn", "epsilon"}:
            raise ValueError("mode must be one of {'knn', 'epsilon'}")
        if not (0 < self._fraction < 1):
            raise ValueError("fraction must satisfy 0 < fraction < 1")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for FaceMethod.fit()")

        with seed_context(self._seed):
            self._feature_groups = resolve_feature_groups(trainset)
            self._feature_names = list(self._feature_groups.feature_names)
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )
            self._train_features = trainset.get(target=False).copy(deep=True)
            self._immutables = list(self._feature_groups.immutable)
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        df = self._train_features.copy(deep=True)
        cond = df.isin(factuals).values
        df = df.drop(df[cond].index)
        df = pd.concat([factuals, df], ignore_index=True)
        df = self._adapter.get_ordered_features(df)

        rows = []
        with seed_context(self._seed):
            for row_index in range(factuals.shape[0]):
                rows.append(
                    graph_search(
                        df,
                        row_index,
                        self._immutables,
                        self._adapter,
                        mode=self._mode,
                        frac=self._fraction,
                    )
                )

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )
