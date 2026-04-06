from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
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


@register("mlp")
class MlpModel(ModelObject):
    def __init__(
        self,
        seed: int | None = None,
        device: str = "cpu",
        epochs: int = 160,
        learning_rate: float = 0.01,
        batch_size: int = 16,
        layers: list[int] | None = None,
        sampling: str = "uniform",
        optimizer: str = "adam",
        criterion: str = "cross_entropy",
        output_activation: str = "softmax",
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
        self._layers: list[int] = list(layers or [32, 16])
        self._sampling: str = str(sampling).lower()
        self._optimizer_name: str = optimizer
        self._criterion_name: str = criterion.lower()
        self._output_activation_name: str = output_activation.lower()
        self._pretrained_path: str | None = pretrained_path
        self._save_name: str | None = save_name
        self._output_dim: int | None = None

        if self._batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self._sampling not in {"uniform", "weighted"}:
            raise ValueError("sampling must be either 'uniform' or 'weighted'")
        if self._criterion_name not in {"cross_entropy", "bce"}:
            raise ValueError(
                "MlpModel supports criterion='cross_entropy' or criterion='bce' only"
            )
        if self._output_activation_name not in {"softmax", "sigmoid"}:
            raise ValueError(
                "MlpModel output_activation must be 'softmax' or 'sigmoid'"
            )
        if self._output_activation_name == "sigmoid" and self._criterion_name != "bce":
            raise ValueError(
                "MlpModel output_activation='sigmoid' requires criterion='bce'"
            )
        if self._output_activation_name == "softmax" and self._criterion_name == "bce":
            raise ValueError(
                "MlpModel output_activation='softmax' is incompatible with criterion='bce'"
            )

    def _build_model(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        blocks: list[torch.nn.Module] = []
        current_dim = input_dim
        for hidden_dim in self._layers:
            blocks.append(torch.nn.Linear(current_dim, hidden_dim))
            blocks.append(torch.nn.ReLU())
            current_dim = hidden_dim
        blocks.append(torch.nn.Linear(current_dim, output_dim))
        return torch.nn.Sequential(*blocks)

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for MlpModel.fit()")

        with seed_context(self._seed):
            X, labels, output_dim = self.extract_training_data(trainset)
            if self._output_activation_name == "sigmoid":
                if len(self.get_class_to_index()) != 2:
                    raise ValueError(
                        "MlpModel output_activation='sigmoid' supports binary classification only"
                    )
                network_output_dim = 1
            else:
                network_output_dim = output_dim
            input_dim = X.shape[1]
            self._output_dim = network_output_dim
            self._model = self._build_model(input_dim, network_output_dim).to(
                self._device
            )

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
            X_tensor = torch.tensor(
                X.to_numpy(dtype="float32"), dtype=torch.float32, device=self._device
            )
            if self._criterion_name == "cross_entropy":
                criterion = torch.nn.CrossEntropyLoss()
                y_tensor = labels.to(self._device)
            else:
                criterion = torch.nn.BCELoss()
                y_tensor = labels.to(self._device).to(dtype=torch.float32).unsqueeze(1)

            sampler = None
            if self._sampling == "weighted":
                label_indices = labels.detach().cpu().numpy()
                class_counts = np.bincount(label_indices)
                if np.any(class_counts == 0):
                    raise ValueError(
                        "Weighted sampling requires every class to appear at least once"
                    )
                class_weights = 1.0 / class_counts
                sample_weights = class_weights[label_indices]
                sampler = WeightedRandomSampler(
                    weights=torch.tensor(sample_weights, dtype=torch.float32),
                    num_samples=X_tensor.shape[0],
                    replacement=True,
                )

            self._model.train()
            for _ in tqdm(range(self._epochs), desc="mlp-fit", leave=False):
                if sampler is None:
                    permutation = torch.randperm(X_tensor.shape[0], device=self._device)
                    batch_index_iterator = (
                        permutation[start : start + self._batch_size]
                        for start in range(0, X_tensor.shape[0], self._batch_size)
                    )
                else:
                    sampled_indices = list(iter(sampler))
                    batch_index_iterator = (
                        torch.tensor(
                            sampled_indices[start : start + self._batch_size],
                            dtype=torch.long,
                            device=self._device,
                        )
                        for start in range(0, len(sampled_indices), self._batch_size)
                    )
                for batch_indices in batch_index_iterator:
                    batch_X = X_tensor[batch_indices]
                    batch_y = y_tensor[batch_indices]
                    optimizer.zero_grad()
                    logits = self._model(batch_X)
                    loss_input = (
                        torch.sigmoid(logits)
                        if self._criterion_name == "bce"
                        else logits
                    )
                    loss = criterion(loss_input, batch_y)
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
                prediction = logits_to_prediction(
                    logits,
                    proba=proba,
                    output_activation=self._output_activation_name,
                )
            return prediction.detach().cpu()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Target model is not trained")
        with seed_context(self._seed):
            self._model.eval()
            logits = self._model(X.to(self._device))
            if self._output_activation_name == "sigmoid":
                if logits.ndim == 1:
                    logits = logits.unsqueeze(1)
                return torch.cat([-logits, logits], dim=1)
            return logits
