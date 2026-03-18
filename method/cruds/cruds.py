from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from dataset.dataset_object import DatasetObject
from method.cchvae.autoencoder import CSVAE
from method.cchvae.cchvae import reconstruct_binary_constraints
from method.cruds.support import (
    RecourseModelAdapter,
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


def compute_loss(cf_initialize, query_instance, target, lambda_param, model_adapter):
    loss_function = nn.BCELoss()
    output = model_adapter.predict_proba(cf_initialize)
    loss1 = loss_function(output, target)
    loss2 = torch.sum((cf_initialize - query_instance) ** 2)
    return loss1 + lambda_param * loss2


def counterfactual_search(
    model_adapter,
    csvae: CSVAE,
    factual: np.ndarray,
    binary_feature_indices: list[int],
    target_class: list[float] | None = None,
    lambda_param: float = 0.001,
    optimizer_name: str = "RMSprop",
    lr: float = 0.008,
    max_iter: int = 2000,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if target_class is None:
        target_class = [0.0, 1.0]

    x_train = factual[:, :-1]
    y_train = np.zeros((factual.shape[0], 2))
    y_train[:, 0] = 1 - factual[:, -1]
    y_train[:, 1] = factual[:, -1]

    _, _, _, _, _, _, _, _, z, _ = csvae.forward(
        torch.from_numpy(x_train).float().to(device),
        torch.from_numpy(y_train).float().to(device),
    )

    w = torch.rand(2, requires_grad=True, dtype=torch.float, device=device)
    target = torch.FloatTensor(np.array(target_class)).to(device)
    target_prediction = int(torch.argmax(target).item())

    if optimizer_name.lower() == "rmsprop":
        optim = torch.optim.RMSprop([w], lr)
    else:
        optim = torch.optim.Adam([w], lr)

    counterfactuals = []
    distances = []
    query_instance = torch.FloatTensor(x_train).to(device)
    z = torch.cat([z, query_instance[:, ~csvae.mutable_mask]], dim=-1)

    for _ in range(max_iter):
        cf, _ = csvae.p_x(z, w.unsqueeze(0))
        temp = query_instance.clone()
        temp[:, csvae.mutable_mask] = cf
        cf = temp
        cf_np = reconstruct_binary_constraints(
            cf.detach().cpu().numpy(),
            binary_feature_indices,
        )
        cf = torch.tensor(cf_np, dtype=torch.float32, device=device)
        output = model_adapter.predict_proba(cf)
        predicted = int(torch.argmax(output[0]).item())

        if predicted == target_prediction:
            counterfactuals.append(
                np.concatenate([cf.detach().cpu().numpy().squeeze(axis=0), [predicted]])
            )

        loss = compute_loss(
            cf, query_instance, target.unsqueeze(0), lambda_param, model_adapter
        )
        if predicted == target_prediction:
            distances.append(float(loss.detach().cpu().item()))
        loss.backward(retain_graph=True)
        optim.step()
        optim.zero_grad()

    if not counterfactuals:
        output = model_adapter.predict_proba(cf)
        predicted = int(torch.argmax(output[0]).item())
        return np.concatenate([cf.detach().cpu().numpy().squeeze(axis=0), [predicted]])

    index = int(np.argmin(np.asarray(distances)))
    return np.asarray(counterfactuals[index])


@register("cruds")
class CrudsMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        lambda_param: float = 0.001,
        optimizer: str = "RMSprop",
        lr: float = 0.008,
        max_iter: int = 2000,
        target_class: list[float] | None = None,
        vae_params: dict | None = None,
        **kwargs,
    ):
        ensure_supported_target_model(
            target_model,
            (LinearModel, MlpModel),
            "CrudsMethod",
        )
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = True
        self._is_trained = False
        self._desired_class = desired_class
        self._lambda_param = float(lambda_param)
        self._optimizer = str(optimizer)
        self._lr = float(lr)
        self._max_iter = int(max_iter)
        self._target_class = [0.0, 1.0] if target_class is None else list(target_class)
        self._vae_params = {
            "layers": [8, 4],
            "train": True,
            "epochs": 20,
            "lr": 1e-3,
            "batch_size": 16,
        }
        self._vae_params.update(vae_params or {})

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for CrudsMethod.fit()")

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
            train_target = trainset.get(target=True)
            train_data = pd.concat([train_features, train_target], axis=1)

            mutable_mask = feature_groups.mutable_mask
            input_dim = int(np.sum(mutable_mask))
            layers = list(self._vae_params["layers"])
            if not layers or layers[0] != input_dim:
                layers = [input_dim] + layers
            self._csvae = CSVAE(
                getattr(trainset, "name", "dataset"),
                layers,
                mutable_mask,
            )
            if self._vae_params["train"]:
                self._csvae.fit(
                    train_data,
                    epochs=int(self._vae_params["epochs"]),
                    lr=float(self._vae_params["lr"]),
                    batch_size=int(self._vae_params["batch_size"]),
                )
            else:
                self._csvae.load(input_dim)
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame):
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        predicted_labels = self._adapter.predict_label_indices(factuals)
        factuals_with_prediction = factuals.copy(deep=True)
        factuals_with_prediction["__label__"] = predicted_labels

        rows = []
        with seed_context(self._seed):
            for _, row in factuals_with_prediction.iterrows():
                rows.append(
                    counterfactual_search(
                        self._adapter,
                        self._csvae,
                        row.to_numpy(dtype="float32").reshape(1, -1),
                        self._binary_indices,
                        target_class=self._target_class,
                        lambda_param=self._lambda_param,
                        optimizer_name=self._optimizer,
                        lr=self._lr,
                        max_iter=self._max_iter,
                    )
                )

        candidates = pd.DataFrame(
            np.asarray(rows)[:, :-1],
            index=factuals.index,
            columns=self._feature_names,
        )
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
