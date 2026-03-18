from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from method.method_object import MethodObject
from method.rbr.rbr_loss import robust_bayesian_recourse
from method.rbr.support import (
    RecourseModelAdapter,
    ensure_supported_target_model,
    validate_counterfactuals,
)
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


class _RbrRawModel:
    def __init__(self, adapter: RecourseModelAdapter):
        self._adapter = adapter

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = self._adapter.predict_proba(x)
        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.tensor(
                probabilities, dtype=torch.float32, device=x.device
            )
        return probabilities


@register("rbr")
class RbrMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        num_samples: int = 200,
        perturb_radius: float = 0.2,
        delta_plus: float = 1.0,
        sigma: float = 1.0,
        epsilon_op: float = 1.0,
        epsilon_pe: float = 1.0,
        max_iter: int = 500,
        clamp: bool = False,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, (MlpModel,), "RbrMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class if desired_class is not None else 1

        self._num_samples = int(num_samples)
        self._perturb_radius = float(perturb_radius)
        self._delta_plus = float(delta_plus)
        self._sigma = float(sigma)
        self._epsilon_op = float(epsilon_op)
        self._epsilon_pe = float(epsilon_pe)
        self._max_iter = int(max_iter)
        self._clamp = bool(clamp)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for RbrMethod.fit()")

        with seed_context(self._seed):
            self._feature_names = list(trainset.get(target=False).columns)
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )
            self._raw_model = _RbrRawModel(self._adapter)
            self._train_data = trainset.get(target=False).to_numpy(dtype=np.float32)
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        rows = []
        with seed_context(self._seed):
            for _, row in factuals.iterrows():
                cf = robust_bayesian_recourse(
                    self._raw_model,
                    row.to_numpy(dtype=np.float32),
                    train_data=self._train_data,
                    num_samples=self._num_samples,
                    perturb_radius=self._perturb_radius,
                    delta_plus=self._delta_plus,
                    sigma=self._sigma,
                    epsilon_op=self._epsilon_op,
                    epsilon_pe=self._epsilon_pe,
                    max_iter=self._max_iter,
                    dev=self._device,
                    random_state=self._seed,
                    verbose=False,
                )
                if self._clamp:
                    cf = np.clip(cf, 0.0, 1.0)
                rows.append(cf)

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
