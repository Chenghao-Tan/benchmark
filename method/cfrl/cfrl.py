from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from dataset.dataset_object import DatasetObject
from method.cfrl.cfrl_backend import set_seed
from method.cfrl.cfrl_tabular import CounterfactualRLTabular as CFRLExplainer
from method.cfrl.cfrl_tabular import get_conditional_dim, get_he_preprocessor
from method.cfrl.support import (
    BlackBoxModelTypes,
    ensure_supported_target_model,
    resolve_feature_groups,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register


class HeterogeneousEncoder(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class HeterogeneousDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dims: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, dim) for dim in output_dims])

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        h = F.relu(self.fc1(z))
        return [head(h) for head in self.heads]


@dataclass
class _FeatureMetadata:
    feature_names: list[str]
    long_to_short: dict[str, str]
    short_to_long: dict[str, str]
    attr_types: dict[str, str]
    attr_bounds: dict[str, list[float]]
    categorical_indices: list[int]
    numerical_indices: list[int]
    raw_to_idx: dict[str, dict[int, int]]
    idx_to_raw: dict[str, dict[int, int]]
    category_map: dict[int, list[str]]
    feature_types: dict[str, type]


@register("cfrl")
class CfrlMethod(MethodObject):
    _DEFAULT_PARAMS: dict[str, object] = {
        "seed": 0,
        "autoencoder_batch_size": 128,
        "autoencoder_target_steps": 2000,
        "autoencoder_lr": 1e-3,
        "autoencoder_latent_dim": 15,
        "autoencoder_hidden_dim": 128,
        "coeff_sparsity": 0.5,
        "coeff_consistency": 0.5,
        "train_steps": 2000,
        "batch_size": 128,
        "immutable_features": None,
        "constrained_ranges": None,
    }

    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        autoencoder_batch_size: int = 128,
        autoencoder_target_steps: int = 2000,
        autoencoder_lr: float = 1e-3,
        autoencoder_latent_dim: int = 15,
        autoencoder_hidden_dim: int = 128,
        coeff_sparsity: float = 0.5,
        coeff_consistency: float = 0.5,
        train_steps: int = 2000,
        batch_size: int = 128,
        immutable_features: list[str] | None = None,
        constrained_ranges: dict[str, list[float]] | None = None,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, BlackBoxModelTypes, "CfrlMethod")
        self._target_model = target_model
        self._seed = seed if seed is not None else int(self._DEFAULT_PARAMS["seed"])
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._params = {
            "seed": self._seed,
            "autoencoder_batch_size": int(autoencoder_batch_size),
            "autoencoder_target_steps": int(autoencoder_target_steps),
            "autoencoder_lr": float(autoencoder_lr),
            "autoencoder_latent_dim": int(autoencoder_latent_dim),
            "autoencoder_hidden_dim": int(autoencoder_hidden_dim),
            "coeff_sparsity": float(coeff_sparsity),
            "coeff_consistency": float(coeff_consistency),
            "train_steps": int(train_steps),
            "batch_size": int(batch_size),
            "immutable_features": immutable_features,
            "constrained_ranges": constrained_ranges,
        }

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")

    def _prepare_metadata(self) -> _FeatureMetadata:
        feature_names = list(self._feature_names)
        long_to_short = {name: name for name in feature_names}
        short_to_long = {name: name for name in feature_names}
        attr_bounds = {name: [0.0, 1.0] for name in feature_names}
        attr_types = {
            name: "numeric-real" if name not in self._binary_feature_names else "binary"
            for name in feature_names
        }
        numerical_indices = list(range(len(feature_names)))
        categorical_indices: list[int] = []
        raw_to_idx: dict[str, dict[int, int]] = {}
        idx_to_raw: dict[str, dict[int, int]] = {}
        category_map: dict[int, list[str]] = {}
        feature_types = {
            name: float if attr_types[name] == "numeric-real" else int
            for name in feature_names
        }
        return _FeatureMetadata(
            feature_names=feature_names,
            long_to_short=long_to_short,
            short_to_long=short_to_long,
            attr_types=attr_types,
            attr_bounds=attr_bounds,
            categorical_indices=categorical_indices,
            numerical_indices=numerical_indices,
            raw_to_idx=raw_to_idx,
            idx_to_raw=idx_to_raw,
            category_map=category_map,
            feature_types=feature_types,
        )

    def _ordered_to_cfrl(self, frame: pd.DataFrame) -> np.ndarray:
        ordered = frame.reindex(columns=self._feature_names, fill_value=0.0)
        return ordered.to_numpy(dtype=np.float32, copy=True)

    def _cfrl_to_ordered(self, arr_zero: np.ndarray) -> pd.DataFrame:
        arr = np.atleast_2d(arr_zero).astype(np.float32, copy=False)
        model_df = pd.DataFrame(arr, columns=self._feature_names)
        return model_df.astype(np.float32)

    def _build_predictor(self):
        def predictor(x: np.ndarray) -> np.ndarray:
            array = np.atleast_2d(x)
            model_input = self._cfrl_to_ordered(array)
            preds = self._target_model.get_prediction(model_input, proba=True)
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()
            elif isinstance(preds, pd.DataFrame):
                preds = preds.to_numpy()
            else:
                preds = np.asarray(preds)
            return preds

        return predictor

    def _train_autoencoder(self, X_pre: np.ndarray, X_zero: np.ndarray) -> None:
        target_steps = int(self._params["autoencoder_target_steps"])
        batch_size = int(self._params["autoencoder_batch_size"])
        lr = float(self._params["autoencoder_lr"])

        inputs = torch.tensor(X_pre, dtype=torch.float32, device=self._device)
        num_dim = len(self._metadata.numerical_indices)
        num_targets = inputs[:, :num_dim]

        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        optimiser = optim.Adam(params, lr=lr)

        num_samples = inputs.size(0)
        steps_per_epoch = max(1, math.ceil(num_samples / batch_size))
        max_epochs = math.ceil(target_steps / steps_per_epoch)
        steps_run = 0

        with tqdm(total=target_steps, desc="cfrl-ae", leave=False) as pbar:
            for _ in range(max_epochs):
                perm = torch.randperm(num_samples, device=self._device)
                for start in range(0, num_samples, batch_size):
                    idx = perm[start : start + batch_size]
                    batch_x = inputs[idx]
                    outputs = self._decoder(self._encoder(batch_x))
                    recon_num = outputs[0]
                    target_num = num_targets[idx]
                    loss = F.mse_loss(recon_num, target_num)
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    steps_run += 1
                    pbar.update(1)
                    if steps_run >= target_steps:
                        break
                if steps_run >= target_steps:
                    break

        self._encoder.eval()
        self._decoder.eval()

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for CfrlMethod.fit()")
        set_seed(int(self._params["seed"]))

        feature_groups = resolve_feature_groups(trainset)
        self._feature_names = list(feature_groups.feature_names)
        self._binary_feature_names = set(feature_groups.binary)
        self._immutable_feature_names = list(feature_groups.immutable)
        self._metadata = self._prepare_metadata()
        self._feature_count = len(self._metadata.feature_names)
        self._raw_feature_set = set(self._metadata.feature_names)

        df_train = trainset.get(target=False)
        X_zero = self._ordered_to_cfrl(df_train).astype(np.float32)
        (
            self._encoder_preprocessor,
            self._decoder_inv_preprocessor,
        ) = get_he_preprocessor(
            X=X_zero,
            feature_names=self._metadata.feature_names,
            category_map=self._metadata.category_map,
            feature_types=self._metadata.feature_types,
        )

        X_pre = self._encoder_preprocessor(X_zero).astype(np.float32)
        latent_dim = int(self._params["autoencoder_latent_dim"])
        hidden_dim = int(self._params["autoencoder_hidden_dim"])

        input_dim = X_pre.shape[1]
        self._encoder = HeterogeneousEncoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            input_dim=input_dim,
        ).to(self._device)
        output_dims = [len(self._metadata.numerical_indices)]
        self._decoder = HeterogeneousDecoder(
            hidden_dim=hidden_dim,
            output_dims=output_dims,
            latent_dim=latent_dim,
        ).to(self._device)

        self._train_autoencoder(X_pre, X_zero)
        predictor = self._build_predictor()

        immutable_features = self._params.get("immutable_features")
        if immutable_features is None:
            immutable_features = list(self._immutable_feature_names)

        ranges = self._params.get("constrained_ranges") or {}
        preds_shape = predictor(X_zero[:1]).shape
        num_classes = preds_shape[1] if len(preds_shape) == 2 else 1
        cond_dim = get_conditional_dim(
            self._metadata.feature_names, self._metadata.category_map
        )
        actor_input_dim = latent_dim + 2 * num_classes + cond_dim

        self._cf_model = CFRLExplainer(
            predictor=predictor,
            encoder=self._encoder,
            decoder=self._decoder,
            latent_dim=latent_dim,
            encoder_preprocessor=self._encoder_preprocessor,
            decoder_inv_preprocessor=self._decoder_inv_preprocessor,
            coeff_sparsity=float(self._params["coeff_sparsity"]),
            coeff_consistency=float(self._params["coeff_consistency"]),
            feature_names=self._metadata.feature_names,
            category_map=self._metadata.category_map,
            immutable_features=immutable_features,
            ranges=ranges,
            train_steps=int(self._params["train_steps"]),
            batch_size=int(self._params["batch_size"]),
            seed=int(self._params["seed"]),
            actor_input_dim=actor_input_dim,
        )
        self._cf_model.fit(X_zero.astype(np.float32))
        self._is_trained = True

    def _generate_counterfactual(self, factual_row: pd.DataFrame) -> pd.DataFrame:
        factual_ordered = factual_row.reindex(
            columns=self._feature_names, fill_value=0.0
        )
        zero_input = self._ordered_to_cfrl(factual_ordered)
        preds = self._target_model.get_prediction(factual_ordered, proba=True)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        target_class = 1 - int(np.argmax(preds, axis=1)[0])

        explanation = self._cf_model.explain(
            X=zero_input.astype(np.float32),
            Y_t=np.array([target_class]),
            C=[],
        )
        cf_data = explanation.get("cf", {}).get("X")
        if cf_data is None:
            return factual_ordered

        cf_array = np.asarray(cf_data)
        if cf_array.ndim == 3:
            cf_array = cf_array[:, 0, :]
        cf_array = np.atleast_2d(cf_array)
        cf_ordered = self._cfrl_to_ordered(cf_array)
        cf_ordered.index = factual_ordered.index
        return cf_ordered

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        set_seed(int(self._params["seed"]))
        factuals = factuals.loc[:, self._feature_names].copy(deep=True)

        results: list[pd.DataFrame] = []
        for index, row in factuals.iterrows():
            cf_row = self._generate_counterfactual(pd.DataFrame([row]))
            cf_row.index = [index]
            results.append(cf_row)

        counterfactuals = pd.concat(results, axis=0)
        return validate_counterfactuals(
            self._target_model,
            factuals,
            counterfactuals,
            desired_class=self._desired_class,
        )
