from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from dataset.dataset_object import DatasetObject
from model.model_object import ModelObject, process_nan
from model.model_utils import (
    build_optimizer,
    logits_to_prediction,
    resolve_device,
    save_torch_model,
)
from utils.registry import register
from utils.seed import seed_context


@register("linear")
class LinearModel(ModelObject):
    def __init__(
        self,
        seed: int | None = None,
        device: str = "cpu",
        epochs: int = 120,
        learning_rate: float = 0.03,
        batch_size: int = 16,
        optimizer: str = "adam",
        criterion: str = "cross_entropy",
        pretrained_path: str | None = None,
        save_name: str | None = None,
        **kwargs,
    ):
        self._model: torch.nn.Module
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = True
        self._is_trained = False
        self._epochs: int = int(epochs)
        self._learning_rate: float = float(learning_rate)
        self._batch_size: int = int(batch_size)
        self._optimizer_name: str = optimizer
        self._criterion_name: str = criterion.lower()
        self._pretrained_path: str | None = pretrained_path
        self._save_name: str | None = save_name
        self._output_dim: int | None = None

        if self._batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self._criterion_name != "cross_entropy":
            raise ValueError(
                "LinearModel currently supports criterion='cross_entropy' only"
            )

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for LinearModel.fit()")

        with seed_context(self._seed):
            X, labels, output_dim = self.extract_training_data(trainset)
            input_dim = X.shape[1]
            self._output_dim = output_dim
            self._model = torch.nn.Linear(input_dim, output_dim).to(self._device)

            if (
                self._pretrained_path is not None
                and Path(self._pretrained_path).exists()
            ):
                state_dict = torch.load(
                    self._pretrained_path, map_location=self._device
                )
                self._model.load_state_dict(state_dict)
                self._model.eval()
                self._is_trained = True
                return

            optimizer = build_optimizer(
                self._optimizer_name, self._model.parameters(), self._learning_rate
            )
            criterion = torch.nn.CrossEntropyLoss()
            X_tensor = torch.tensor(
                X.to_numpy(dtype="float32"), dtype=torch.float32, device=self._device
            )
            y_tensor = labels.to(self._device)

            self._model.train()
            for _ in tqdm(range(self._epochs), desc="linear-fit", leave=False):
                permutation = torch.randperm(X_tensor.shape[0], device=self._device)
                for start in range(0, X_tensor.shape[0], self._batch_size):
                    batch_indices = permutation[start : start + self._batch_size]
                    batch_X = X_tensor[batch_indices]
                    batch_y = y_tensor[batch_indices]
                    optimizer.zero_grad()
                    logits = self._model(batch_X)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

            self._model.eval()
            self._is_trained = True
            save_torch_model(self._model, self._save_name)

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Target model is not trained")
        with seed_context(self._seed):
            self._model.eval()
            X_tensor = torch.tensor(
                X.to_numpy(dtype="float32"), dtype=torch.float32, device=self._device
            )
            with torch.no_grad():
                logits = self._model(X_tensor)
                prediction = logits_to_prediction(logits, proba=proba)
            return prediction.detach().cpu()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Target model is not trained")
        with seed_context(self._seed):
            self._model.eval()
            return self._model(X.to(self._device))
