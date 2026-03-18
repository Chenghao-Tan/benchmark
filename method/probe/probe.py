from __future__ import annotations

import pandas as pd

from dataset.dataset_object import DatasetObject
from method.method_object import MethodObject
from method.probe.search import probe_recourse
from method.probe.support import (
    RecourseModelAdapter,
    TorchModelTypes,
    ensure_supported_target_model,
    resolve_feature_groups,
    validate_counterfactuals,
)
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("probe")
class ProbeMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        feature_cost: str | None = None,
        lr: float = 0.001,
        lambda_: float = 0.01,
        n_iter: int = 1000,
        t_max_min: float = 1.0,
        norm: int = 1,
        clamp: bool = True,
        loss_type: str = "MSE",
        y_target: list[float] | None = None,
        noise_variance: float = 0.01,
        invalidation_target: float = 0.45,
        inval_target_eps: float = 0.005,
        target_class: int | str | None = None,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, TorchModelTypes, "ProbeMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = True
        self._is_trained = False
        self._desired_class = (
            desired_class if desired_class is not None else target_class
        )
        self._feature_cost = feature_cost
        self._lr = float(lr)
        self._lambda = float(lambda_)
        self._n_iter = int(n_iter)
        self._t_max_min = float(t_max_min)
        self._norm = int(norm)
        self._clamp = bool(clamp)
        self._loss_type = str(loss_type).upper()
        self._y_target = None if y_target is None else list(y_target)
        self._noise_variance = float(noise_variance)
        self._invalidation_target = float(invalidation_target)
        self._inval_target_eps = float(inval_target_eps)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ProbeMethod.fit()")

        with seed_context(self._seed):
            feature_groups = resolve_feature_groups(trainset)
            self._feature_names = list(feature_groups.feature_names)
            self._binary_indices = [
                self._feature_names.index(feature_name)
                for feature_name in feature_groups.binary
            ]
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        class_to_index = self._target_model.get_class_to_index()
        if self._desired_class is None and self._y_target is None:
            desired_index = 1
            desired_class = next(
                class_value
                for class_value, class_index in class_to_index.items()
                if class_index == desired_index
            )
        else:
            desired_class = self._desired_class
            desired_index = class_to_index[desired_class]
        y_target = self._y_target
        if y_target is None:
            y_target = [0.0] * len(class_to_index)
            y_target[desired_index] = 1.0

        rows = []
        with seed_context(self._seed):
            for _, row in factuals.iterrows():
                rows.append(
                    probe_recourse(
                        model=self._adapter,
                        x=row.to_numpy(dtype="float32"),
                        binary_feature_indices=self._binary_indices,
                        lr=self._lr,
                        lambda_param=self._lambda,
                        y_target=y_target,
                        n_iter=self._n_iter,
                        t_max_min=self._t_max_min,
                        norm=self._norm,
                        clamp=self._clamp,
                        loss_type=self._loss_type,
                        invalidation_target=self._invalidation_target,
                        inval_target_eps=self._inval_target_eps,
                        noise_variance=self._noise_variance,
                    )
                )

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=desired_class,
        )
