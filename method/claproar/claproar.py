from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dataset_object import DatasetObject
from method.claproar.support import (
    RecourseModelAdapter,
    TorchModelTypes,
    ensure_supported_target_model,
    resolve_target_indices,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("claproar")
class ClaProarMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        individual_cost_lambda: float = 0.1,
        external_cost_lambda: float = 0.1,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-4,
        target_class: int | str | None = None,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, TorchModelTypes, "ClaProarMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = True
        self._is_trained = False
        self._desired_class = (
            desired_class if desired_class is not None else target_class
        )

        self._individual_cost_lambda = float(individual_cost_lambda)
        self._external_cost_lambda = float(external_cost_lambda)
        self._learning_rate = float(learning_rate)
        self._max_iter = int(max_iter)
        self._tol = float(tol)
        self._criterion = nn.CrossEntropyLoss()

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self._max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self._tol <= 0:
            raise ValueError("tol must be > 0")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ClaProarMethod.fit()")

        with seed_context(self._seed):
            self._feature_names = list(trainset.get(target=False).columns)
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )
            self._is_trained = True

    def _compute_costs(
        self,
        original: torch.Tensor,
        candidate: torch.Tensor,
        target_index: int,
    ) -> torch.Tensor:
        logits = self._target_model.forward(candidate.unsqueeze(0))
        target_tensor = torch.tensor(
            [target_index], dtype=torch.long, device=self._device
        )
        external_target = torch.tensor(
            [1 - target_index], dtype=torch.long, device=self._device
        )
        yloss = self._criterion(logits, target_tensor)
        external_cost = self._criterion(logits, external_target)
        individual_cost = torch.norm(original - candidate, p=2)
        return (
            yloss
            + self._individual_cost_lambda * individual_cost
            + self._external_cost_lambda * external_cost
        )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        original_prediction = self._adapter.predict_label_indices(factuals)
        target_indices = resolve_target_indices(
            self._target_model,
            original_prediction,
            self._desired_class,
        )

        rows = []
        with seed_context(self._seed):
            for row_index, (_, row) in enumerate(factuals.iterrows()):
                original = torch.tensor(
                    row.to_numpy(dtype="float32"),
                    dtype=torch.float32,
                    device=self._device,
                )
                candidate = original.clone().detach().requires_grad_(True)
                optimizer_cf = optim.Adam([candidate], lr=self._learning_rate)

                for _ in range(self._max_iter):
                    optimizer_cf.zero_grad()
                    objective = self._compute_costs(
                        original=original,
                        candidate=candidate,
                        target_index=int(target_indices[row_index]),
                    )
                    objective.backward()
                    optimizer_cf.step()
                    if (
                        candidate.grad is not None
                        and torch.norm(candidate.grad) < self._tol
                    ):
                        break

                rows.append(
                    pd.Series(
                        candidate.detach().cpu().numpy(),
                        index=self._feature_names,
                    )
                )

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
