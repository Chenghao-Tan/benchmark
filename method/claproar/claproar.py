from __future__ import annotations

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
            features = trainset.get(target=False)
            try:
                features.to_numpy(dtype="float32")
            except ValueError as error:
                raise ValueError(
                    "ClaProarMethod requires fully numeric input features"
                ) from error

            class_to_index = self._target_model.get_class_to_index()
            if len(class_to_index) != 2:
                raise ValueError(
                    "ClaProarMethod currently supports binary classification only"
                )
            if (
                self._desired_class is not None
                and self._desired_class not in class_to_index
            ):
                raise ValueError(
                    "desired_class is invalid for the trained target model"
                )

            self._feature_names = list(features.columns)
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
        probabilities = self._adapter.predict_proba(candidate.unsqueeze(0))
        if not isinstance(probabilities, torch.Tensor):
            raise RuntimeError("ClaProarMethod requires differentiable probabilities")
        target_tensor = torch.tensor(
            [target_index], dtype=torch.long, device=self._device
        )
        external_target = torch.tensor(
            [1 - target_index], dtype=torch.long, device=self._device
        )
        yloss = self._criterion(probabilities, target_tensor)
        external_cost = self._criterion(probabilities, external_target)
        individual_cost = torch.norm(original - candidate, p=2)
        return (
            yloss
            + self._individual_cost_lambda * individual_cost
            + self._external_cost_lambda * external_cost
        )

    def get_counterfactuals(
        self, factuals: pd.DataFrame, raw_output: bool = False
    ) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        try:
            factuals.to_numpy(dtype="float32")
        except ValueError as error:
            raise ValueError(
                "ClaProarMethod requires fully numeric input features"
            ) from error

        original_prediction = self._adapter.predict_label_indices(factuals)
        target_indices = resolve_target_indices(
            self._target_model,
            original_prediction,
            self._desired_class,
        )
        if factuals.shape[0] == 0:
            return factuals.copy(deep=True)

        factual_array = factuals.to_numpy(dtype="float32")
        original = torch.tensor(
            factual_array,
            dtype=torch.float32,
            device=self._device,
        )
        target_tensor = torch.tensor(
            target_indices,
            dtype=torch.long,
            device=self._device,
        )
        active_mask = torch.tensor(
            original_prediction != target_indices,
            dtype=torch.bool,
            device=self._device,
        )
        candidate = original.clone().detach().requires_grad_(True)
        with seed_context(self._seed):
            if bool(active_mask.any()):
                optimizer_cf = optim.Adam([candidate], lr=self._learning_rate)
                for _ in range(self._max_iter):
                    optimizer_cf.zero_grad()
                    probabilities = self._adapter.predict_proba(candidate)
                    if not isinstance(probabilities, torch.Tensor):
                        raise RuntimeError(
                            "ClaProarMethod requires differentiable probabilities"
                        )
                    yloss = self._criterion(
                        probabilities[active_mask],
                        target_tensor[active_mask],
                    )
                    external_targets = 1 - target_tensor[active_mask]
                    external_cost = self._criterion(
                        probabilities[active_mask],
                        external_targets,
                    )
                    individual_cost = torch.linalg.vector_norm(
                        original[active_mask] - candidate[active_mask],
                        ord=2,
                        dim=1,
                    ).mean()
                    objective = (
                        yloss
                        + self._individual_cost_lambda * individual_cost
                        + self._external_cost_lambda * external_cost
                    )
                    objective.backward()
                    grad_norm = (
                        None
                        if candidate.grad is None
                        else torch.linalg.vector_norm(candidate.grad[active_mask]).item()
                    )
                    optimizer_cf.step()
                    if grad_norm is not None and grad_norm < self._tol:
                        break

        final_candidates = candidate.detach()
        if bool((~active_mask).any()):
            final_candidates[~active_mask] = original[~active_mask]
        candidates = pd.DataFrame(
            final_candidates.cpu().numpy(),
            index=factuals.index,
            columns=self._feature_names,
        )
        if raw_output:
            return candidates
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
