from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from numpy import linalg as LA

from dataset.dataset_object import DatasetObject
from method.cchvae.autoencoder import VariationalAutoencoder
from method.cchvae.support import (
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


def reconstruct_binary_constraints(
    x: np.ndarray,
    binary_feature_indices: list[int],
) -> np.ndarray:
    if not binary_feature_indices:
        return x
    output = x.copy()
    output[:, binary_feature_indices] = np.clip(
        np.round(output[:, binary_feature_indices]), 0.0, 1.0
    )
    return output


@register("cchvae")
class CchvaeMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        n_search_samples: int = 300,
        p_norm: int = 1,
        step: float = 0.1,
        max_iter: int = 1000,
        clamp: bool = True,
        vae_params: dict | None = None,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, BlackBoxModelTypes, "CchvaeMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class
        self._n_search_samples = int(n_search_samples)
        self._p_norm = int(p_norm)
        self._step = float(step)
        self._max_iter = int(max_iter)
        self._clamp = bool(clamp)
        self._vae_params = {
            "layers": [8, 4],
            "train": True,
            "kl_weight": 0.3,
            "lambda_reg": 1e-6,
            "epochs": 20,
            "lr": 1e-3,
            "batch_size": 16,
        }
        self._vae_params.update(vae_params or {})

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for CchvaeMethod.fit()")

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
            self._generative_model = VariationalAutoencoder(
                getattr(trainset, "name", "dataset"),
                layers,
                mutable_mask,
            )
            if self._vae_params["train"]:
                self._generative_model.fit(
                    xtrain=train_features,
                    kl_weight=float(self._vae_params["kl_weight"]),
                    lambda_reg=float(self._vae_params["lambda_reg"]),
                    epochs=int(self._vae_params["epochs"]),
                    lr=float(self._vae_params["lr"]),
                    batch_size=int(self._vae_params["batch_size"]),
                )
            else:
                self._generative_model.load(input_dim)
            self._is_trained = True

    def _hyper_sphere_coordinates(self, instance, high: int, low: int):
        delta_instance = np.random.randn(self._n_search_samples, instance.shape[1])
        dist = np.random.rand(self._n_search_samples) * (high - low) + low
        norm_p = LA.norm(delta_instance, ord=self._p_norm, axis=1)
        d_norm = np.divide(dist, norm_p).reshape(-1, 1)
        delta_instance = np.multiply(delta_instance, d_norm)
        return instance + delta_instance, dist

    def _counterfactual_search(self, factual: np.ndarray) -> np.ndarray:
        low = 0
        high = self._step
        count = 0
        device = self._generative_model.device

        factual_array = factual.reshape(1, -1).astype(np.float32)
        factual_tensor = torch.from_numpy(factual_array).to(device)
        instance_label = np.argmax(self._adapter.predict_proba(factual_array), axis=1)

        z = self._generative_model.encode(
            factual_tensor[:, self._generative_model.mutable_mask].float()
        )[0]
        z = torch.cat(
            [z, factual_tensor[:, ~self._generative_model.mutable_mask]], dim=-1
        )
        z = z.cpu().detach().numpy()
        z_rep = np.repeat(z.reshape(1, -1), self._n_search_samples, axis=0)
        fact_rep = factual_tensor.reshape(1, -1).repeat_interleave(
            self._n_search_samples, dim=0
        )

        x_ce = np.full((1, len(self._feature_names)), np.nan, dtype=np.float32)
        candidate_dist: list[float] = []
        while count <= self._max_iter or len(candidate_dist) <= 0:
            count += 1
            if count > self._max_iter:
                return x_ce[0]

            latent_neighbourhood, _ = self._hyper_sphere_coordinates(z_rep, high, low)
            torch_latent_neighbourhood = (
                torch.from_numpy(latent_neighbourhood).to(device).float()
            )
            decoded = self._generative_model.decode(torch_latent_neighbourhood)
            temp = fact_rep.clone()
            temp[:, self._generative_model.mutable_mask] = decoded.to(temp.dtype)
            x_ce_tensor = temp

            x_ce = x_ce_tensor.detach().cpu().numpy()
            x_ce = reconstruct_binary_constraints(x_ce, self._binary_indices)
            x_ce = x_ce.clip(0, 1) if self._clamp else x_ce

            if self._p_norm == 1:
                distances = np.abs((x_ce - factual_array)).sum(axis=1)
            else:
                distances = LA.norm(x_ce - factual_array, axis=1)

            y_candidate = np.argmax(self._adapter.predict_proba(x_ce), axis=1)
            indices = np.where(y_candidate != instance_label)
            candidate_counterfactuals = x_ce[indices]
            candidate_dist = distances[indices]
            if len(candidate_dist) == 0:
                low = high
                high = low + self._step
            else:
                min_index = np.argmin(candidate_dist)
                return candidate_counterfactuals[min_index]

        return x_ce[0]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        rows = []
        with seed_context(self._seed):
            for _, row in factuals.iterrows():
                rows.append(self._counterfactual_search(row.to_numpy(dtype="float32")))

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
