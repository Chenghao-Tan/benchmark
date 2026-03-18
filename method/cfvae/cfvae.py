from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from dataset.dataset_object import DatasetObject
from method.cfvae.support import (
    ensure_supported_target_model,
    resolve_feature_groups,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


def _set_seed(seed: int = 10_000_000) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _CFVAE(nn.Module):
    def __init__(self, feature_num: int, encoded_size: int):
        super().__init__()
        self.feature_num = feature_num
        self.encoded_size = encoded_size

        self.encoder_mean = nn.Sequential(
            nn.Linear(self.feature_num + 1, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size),
        )

        self.encoder_var = nn.Sequential(
            nn.Linear(self.feature_num + 1, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size),
            nn.Sigmoid(),
        )

        self.decoder_mean = nn.Sequential(
            nn.Linear(self.encoded_size + 1, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, self.feature_num),
            nn.Sigmoid(),
        )

    def encoder(self, x: torch.Tensor):
        x = x.float()
        mean = self.encoder_mean(x)
        var = 0.5 + self.encoder_var(x)
        return mean, var

    def sample_latent_code(self, mean: torch.Tensor, var: torch.Tensor):
        mean, var = mean.float(), var.float()
        eps = torch.randn_like(var)
        return mean + torch.sqrt(var) * eps

    def decoder(self, z: torch.Tensor):
        z = z.float()
        return self.decoder_mean(z)

    def forward(self, x: torch.Tensor, conditions: torch.Tensor, sample: bool = True):
        x, conditions = x.float(), conditions.view(-1, 1).float()
        mean, var = self.encoder(torch.cat((x, conditions), dim=1))
        z = self.sample_latent_code(mean, var) if sample else mean
        return self.decoder(torch.cat((z, conditions), dim=1))

    def forward_with_kl(
        self, x: torch.Tensor, conditions: torch.Tensor, sample: bool = True
    ):
        x, conditions = x.float(), conditions.view(-1, 1).float()
        mean, var = self.encoder(torch.cat((x, conditions), dim=1))
        kl_divergence = 0.5 * torch.mean(mean**2 + var - torch.log(var) - 1, dim=1)
        z = self.sample_latent_code(mean, var) if sample else mean
        x_pred = self.decoder(torch.cat((z, conditions), 1))
        return x_pred, kl_divergence


@register("cfvae")
class CfvaeMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        encoded_size: int = 10,
        train: bool = True,
        save_path: str | None = None,
        batch_size: int = 128,
        epoch: int = 30,
        learning_rate: float = 1e-2,
        n_samples: int = 20,
        margin: float = 0.2,
        validity_reg: float = 20.0,
        **kwargs,
    ):
        ensure_supported_target_model(
            target_model,
            (LinearModel, MlpModel),
            "CfvaeMethod",
        )
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = True
        self._is_trained = False
        self._desired_class = desired_class if desired_class is not None else 1

        self._encoded_size = int(encoded_size)
        self._train_when_fit = bool(train)
        self._save_path = save_path
        self._batch_size = int(batch_size)
        self._epoch = int(epoch)
        self._learning_rate = float(learning_rate)
        self._n_samples = int(n_samples)
        self._margin = float(margin)
        self._validity_reg = float(validity_reg)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for CfvaeMethod.fit()")

        with seed_context(self._seed):
            feature_groups = resolve_feature_groups(trainset)
            self._feature_names = list(feature_groups.feature_names)
            self._binary_indices = [
                self._feature_names.index(feature_name)
                for feature_name in feature_groups.binary
            ]
            train_features = trainset.get(target=False).copy(deep=True)
            self._cf_model = _CFVAE(len(self._feature_names), self._encoded_size).to(
                self._device
            )
            if self._train_when_fit:
                self._train_cfvae(train_features)
            self._is_trained = True

    def _train_cfvae(self, train_features: pd.DataFrame):
        _set_seed()
        train_tensor = torch.tensor(
            train_features.values, dtype=torch.float32, device=self._device
        )
        dataset_size = train_tensor.size(0)
        train_loader = torch.utils.data.DataLoader(
            train_tensor,
            batch_size=self._batch_size,
            shuffle=True,
            pin_memory=False,
        )

        optimizer = optim.SGD(
            self._cf_model.parameters(),
            lr=self._learning_rate,
            weight_decay=1e-2,
        )
        self._cf_model.train()

        for _ in tqdm(range(self._epoch), desc="cfvae-fit", leave=False):
            with tqdm(total=dataset_size, desc="loss: N/A", leave=False) as pbar:
                for train_x in train_loader:
                    train_x = train_x.float().to(self._device)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        train_y = 1.0 - torch.argmax(
                            self._target_model.forward(train_x), dim=1
                        )

                    reconstruction_loss = torch.zeros(
                        train_x.size(0), device=self._device
                    )
                    kl_loss = torch.zeros(train_x.size(0), device=self._device)
                    validity_loss = torch.zeros(1, device=self._device)

                    for _ in range(self._n_samples):
                        x_pred, kl_loss = self._cf_model.forward_with_kl(
                            train_x, train_y
                        )
                        reconstruction_loss += -torch.sum(
                            torch.abs(train_x - x_pred), dim=1
                        )
                        y_pred = self._target_model.forward(x_pred)
                        y_pred_pos = y_pred[train_y == 1, :]
                        y_pred_neg = y_pred[train_y == 0, :]
                        if torch.sum(train_y == 1) > 0:
                            validity_loss += F.hinge_embedding_loss(
                                y_pred_pos[:, 1] - y_pred_pos[:, 0],
                                torch.tensor(-1, device=self._device),
                                self._margin,
                                reduction="mean",
                            )
                        if torch.sum(train_y == 0) > 0:
                            validity_loss += F.hinge_embedding_loss(
                                y_pred_neg[:, 0] - y_pred_neg[:, 1],
                                torch.tensor(-1, device=self._device),
                                self._margin,
                                reduction="mean",
                            )

                    reconstruction_loss = reconstruction_loss / self._n_samples
                    kl_loss = kl_loss / self._n_samples
                    validity_loss = (
                        -1.0 * self._validity_reg * validity_loss / self._n_samples
                    )

                    loss = -torch.mean(reconstruction_loss - kl_loss) - validity_loss
                    loss.backward()
                    optimizer.step()
                    pbar.update(len(train_x))

        if self._save_path:
            Path(self._save_path).mkdir(parents=True, exist_ok=True)
            path = os.path.join(self._save_path, "cfvae_model.pth")
            torch.save(self._cf_model.state_dict(), path)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        _set_seed()
        self._cf_model.eval().to(self._device)
        factuals = factuals.loc[:, self._feature_names].copy(deep=True)

        rows = []
        with seed_context(self._seed):
            for _, row in factuals.iterrows():
                with torch.no_grad():
                    test_x = torch.tensor(
                        row.to_numpy(dtype="float32"), device=self._device
                    ).view(1, -1)
                    test_y = 1.0 - torch.argmax(
                        self._target_model.forward(test_x), dim=1
                    )
                    x_pred = self._cf_model(test_x, test_y, sample=False)
                    if self._binary_indices:
                        x_pred[:, self._binary_indices] = torch.round(
                            x_pred[:, self._binary_indices]
                        )
                    rows.append(x_pred.view(-1).cpu().numpy())

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
