from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from dataset.dataset_object import DatasetObject
from method.cchvae.autoencoder import VariationalAutoencoder
from method.cchvae.cchvae import reconstruct_binary_constraints
from method.method_object import MethodObject
from method.revise.support import (
    RecourseModelAdapter,
    ensure_supported_target_model,
    resolve_feature_groups,
    validate_counterfactuals,
)
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("revise")
class ReviseMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        lambda_: float = 0.5,
        optimizer: str = "adam",
        lr: float = 0.1,
        max_iter: int = 1000,
        target_class: list[float] | None = None,
        vae_params: dict | None = None,
        **kwargs,
    ):
        ensure_supported_target_model(
            target_model,
            (LinearModel, MlpModel),
            "ReviseMethod",
        )
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = True
        self._is_trained = False
        self._desired_class = desired_class
        self._lambda = float(lambda_)
        self._optimizer = str(optimizer).lower()
        self._lr = float(lr)
        self._max_iter = int(max_iter)
        self._target_class = [0.0, 1.0] if target_class is None else list(target_class)
        self._vae_params = {
            "layers": [8, 4],
            "train": True,
            "lambda_reg": 1e-6,
            "epochs": 20,
            "lr": 1e-3,
            "batch_size": 16,
        }
        self._vae_params.update(vae_params or {})

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._optimizer not in {"adam", "rmsprop"}:
            raise ValueError("optimizer must be one of {'adam', 'rmsprop'}")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ReviseMethod.fit()")

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
            train_features = trainset.get(target=False)
            mutable_mask = feature_groups.mutable_mask
            input_dim = int(np.sum(mutable_mask))
            layers = list(self._vae_params["layers"])
            if not layers or layers[0] != input_dim:
                layers = [input_dim] + layers
            self._vae = VariationalAutoencoder(
                getattr(trainset, "name", "dataset"),
                layers,
                mutable_mask,
            )
            if self._vae_params["train"]:
                self._vae.fit(
                    xtrain=train_features,
                    lambda_reg=float(self._vae_params["lambda_reg"]),
                    epochs=int(self._vae_params["epochs"]),
                    lr=float(self._vae_params["lr"]),
                    batch_size=int(self._vae_params["batch_size"]),
                )
            else:
                self._vae.load(input_dim)
            self._is_trained = True

    def _compute_loss(
        self,
        probabilities: torch.Tensor,
        cf_initialize: torch.Tensor,
        query_instance: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        loss_function = nn.BCELoss()
        loss1 = loss_function(probabilities, target)
        loss2 = torch.norm((cf_initialize - query_instance), 1)
        return loss1 + self._lambda * loss2

    def _counterfactual_optimization(self, df_fact: pd.DataFrame):
        device = self._device
        test_loader = torch.utils.data.DataLoader(
            df_fact.values, batch_size=1, shuffle=False
        )
        mutable_mask_tensor = torch.tensor(
            self._vae.mutable_mask, dtype=torch.bool, device=device
        )
        mutable_indices = torch.nonzero(mutable_mask_tensor, as_tuple=False).squeeze(1)
        mutable_indices = (
            mutable_indices if mutable_indices.ndim else mutable_indices.unsqueeze(0)
        )

        list_cfs = []
        for query_instance in test_loader:
            query_instance = query_instance.float().to(device)
            target = torch.tensor(
                self._target_class, dtype=torch.float32, device=device
            )
            target_prediction = int(torch.argmax(target).item())

            z = self._vae.encode(query_instance[:, self._vae.mutable_mask])[0]
            z = torch.cat([z, query_instance[:, ~self._vae.mutable_mask]], dim=-1)
            z = z.clone().detach().requires_grad_(True)
            if self._optimizer == "adam":
                optim = torch.optim.Adam([z], self._lr)
            else:
                optim = torch.optim.RMSprop([z], self._lr)

            candidate_counterfactuals = []
            candidate_distances = []
            for _ in range(self._max_iter):
                decoded_cf = self._vae.decode(z)
                index = mutable_indices.unsqueeze(0).expand(query_instance.size(0), -1)
                cf = query_instance.scatter(
                    1, index, decoded_cf.to(query_instance.dtype)
                )
                cf_hard = cf
                cf_hard_np = reconstruct_binary_constraints(
                    cf.detach().cpu().numpy(),
                    self._binary_indices,
                )
                cf_hard = torch.tensor(cf_hard_np, dtype=torch.float32, device=device)
                output_soft = self._adapter.predict_proba(cf)
                output_hard = self._adapter.predict_proba(cf_hard)
                predicted = int(torch.argmax(output_hard[0]).item())
                loss = self._compute_loss(output_soft[0], cf, query_instance, target)

                if predicted == target_prediction:
                    candidate_counterfactuals.append(
                        cf_hard.detach().cpu().numpy().squeeze(axis=0)
                    )
                    candidate_distances.append(float(loss.detach().cpu().item()))

                loss.backward()
                optim.step()
                optim.zero_grad()

            if len(candidate_counterfactuals):
                array_counterfactuals = np.array(candidate_counterfactuals)
                array_distances = np.array(candidate_distances)
                index = int(np.argmin(array_distances))
                list_cfs.append(array_counterfactuals[index])
            else:
                cf_tensor = reconstruct_binary_constraints(
                    query_instance.detach().cpu().numpy(),
                    self._binary_indices,
                )
                list_cfs.append(cf_tensor.squeeze(axis=0))
        return list_cfs

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        rows = self._counterfactual_optimization(factuals)
        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
