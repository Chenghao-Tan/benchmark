from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dataset_object import DatasetObject
from method.gravitational.support import (
    RecourseModelAdapter,
    TorchModelTypes,
    ensure_supported_target_model,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("gravitational")
class GravitationalMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        prediction_loss_lambda: float = 1.0,
        original_dist_lambda: float = 0.5,
        grav_penalty_lambda: float = 1.5,
        learning_rate: float = 0.01,
        num_steps: int = 100,
        target_class: int | str | None = 1,
        scheduler_step_size: int = 100,
        scheduler_gamma: float = 0.5,
        **kwargs,
    ):
        ensure_supported_target_model(
            target_model, TorchModelTypes, "GravitationalMethod"
        )
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = True
        self._is_trained = False
        self._desired_class = (
            desired_class if desired_class is not None else target_class
        )

        self._prediction_loss_lambda = float(prediction_loss_lambda)
        self._original_dist_lambda = float(original_dist_lambda)
        self._grav_penalty_lambda = float(grav_penalty_lambda)
        self._learning_rate = float(learning_rate)
        self._num_steps = int(num_steps)
        self._scheduler_step_size = int(scheduler_step_size)
        self._scheduler_gamma = float(scheduler_gamma)
        self._criterion = nn.CrossEntropyLoss()

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self._num_steps < 1:
            raise ValueError("num_steps must be >= 1")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for GravitationalMethod.fit()")

        with seed_context(self._seed):
            self._feature_names = list(trainset.get(target=False).columns)
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )

            train_features = trainset.get(target=False)
            train_target = trainset.get(target=True).iloc[:, 0]
            class_to_index = self._target_model.get_class_to_index()
            if self._desired_class not in class_to_index:
                raise ValueError(
                    "desired_class is invalid for the trained target model"
                )
            desired_index = int(class_to_index[self._desired_class])

            mapped_target = train_target.map(class_to_index).astype(int)
            mask = mapped_target == desired_index
            if mask.any():
                x_center = train_features.loc[mask.to_numpy()].mean(axis=0)
            else:
                x_center = train_features.mean(axis=0)
            self._x_center = torch.tensor(
                np.nan_to_num(x_center.to_numpy(dtype="float32"), nan=0.0),
                dtype=torch.float32,
                device=self._device,
            )
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        class_to_index = self._target_model.get_class_to_index()
        desired_index = int(class_to_index[self._desired_class])

        rows = []
        with seed_context(self._seed):
            for _, row in factuals.iterrows():
                original = torch.tensor(
                    row.to_numpy(dtype="float32"),
                    dtype=torch.float32,
                    device=self._device,
                )
                candidate = original.clone().detach().requires_grad_(True)
                optimizer = optim.Adam([candidate], lr=self._learning_rate)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self._scheduler_step_size,
                    gamma=self._scheduler_gamma,
                )

                for _ in range(self._num_steps):
                    optimizer.zero_grad()
                    logits = self._target_model.forward(candidate.unsqueeze(0))
                    target_tensor = torch.tensor(
                        [desired_index], dtype=torch.long, device=self._device
                    )
                    prediction_loss = self._criterion(logits, target_tensor)
                    original_dist = torch.norm(original - candidate, p=2)
                    grav_penalty = torch.norm(candidate - self._x_center, p=2)
                    loss = (
                        self._prediction_loss_lambda * prediction_loss
                        + self._original_dist_lambda * original_dist
                        + self._grav_penalty_lambda * grav_penalty
                    )
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

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
