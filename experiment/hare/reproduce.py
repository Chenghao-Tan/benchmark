from __future__ import annotations

import argparse
import datetime
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from scipy.spatial.distance import cosine as scipy_cosine
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiment import Experiment
from method.hare.oracles import select_best_candidate
from method.hare.sampling import actionable_sampling
from method.hare.search import boundary_point_search, calibrate_candidate_set
from method.hare.support import (
    FeatureSchema,
    resolve_feature_schema,
    validate_counterfactuals,
)
from model.model_object import ModelObject
from utils.seed import seed_context

DEFAULT_ANN_CONFIG = Path("./experiment/hare/compas_wachter_ann_reproduce.yaml")
TABLE1_TARGETS = {
    "base": {"gtp": 0.39, "gtd": 0.75},
    "random": {"gtp": 0.39, "gtd": 0.87},
    "hare": {"gtp": 0.27, "gtd": 0.27},
    "multi_hare": {"gtp": 0.27, "gtd": 0.26},
}

TABLE4_ANN_TARGETS = {
    "base": {
        "gtp": 0.39,
        "gtd": 0.75,
        "success_rate": 0.83,
        "constraint_violation": 0.00,
        "redundancy": 2.02,
        "proximity": 0.03,
        "sparsity": 2.95,
    },
    "hare": {
        "gtp": 0.27,
        "gtd": 0.27,
        "success_rate": 0.82,
        "constraint_violation": 0.00,
        "redundancy": 1.67,
        "proximity": 0.09,
        "sparsity": 3.00,
    },
    "multi_hare": {
        "gtp": 0.27,
        "gtd": 0.26,
        "success_rate": 0.82,
        "constraint_violation": 0.00,
        "redundancy": 1.74,
        "proximity": 0.09,
        "sparsity": 2.99,
    },
}

VARIANT_LABELS = {
    "base": "Base",
    "random": "Random",
    "hare": "HARE",
    "multi_hare": "Multi-HARE",
}
REFERENCE_COMPAS_SPLIT_URLS = {
    "train": "https://raw.githubusercontent.com/carla-recourse/cf-data/master/compas_train.csv",
    "test": "https://raw.githubusercontent.com/carla-recourse/cf-data/master/compas_test.csv",
}
CARLA_REFERENCE_MODEL_DEFAULTS = {
    "epochs": 20,
    "learning_rate": 0.002,
    "batch_size": 256,
    "layers": [10, 5, 10],
    "optimizer": "rms",
    "criterion": "bce",
    "output_activation": "softmax",
}
CARLA_REFERENCE_WACHTER_DEFAULTS = {
    "baseline_lr": 0.01,
    "baseline_lambda": 0.1,
    "baseline_n_iter": 2500,
    "baseline_t_max_min": 0.7,
    "baseline_norm": 1,
    "baseline_loss_type": "BCE",
}
DECISION_THRESHOLD = 0.5

PAPER_COMPAS_TARGET = "score"
PAPER_COMPAS_FEATURE_ORDER = [
    "age",
    "two_year_recid",
    "priors_count",
    "length_of_stay",
    "c_charge_degree_M",
    "race_Other",
    "sex_Male",
]
PAPER_COMPAS_DF_ORDER = PAPER_COMPAS_FEATURE_ORDER + [PAPER_COMPAS_TARGET]
PAPER_COMPAS_SOURCE_COLUMNS = {
    "age": "age",
    "two_year_recid": "two_year_recid",
    "priors_count": "priors_count",
    "length_of_stay": "length_of_stay",
    "c_charge_degree_M": "c_charge_degree_cat_M",
    "race_Other": "race_cat_Other",
    "sex_Male": "sex_cat_Male",
    PAPER_COMPAS_TARGET: PAPER_COMPAS_TARGET,
}
PAPER_COMPAS_ENCODING = {
    "c_charge_degree": ["c_charge_degree_M"],
    "race": ["race_Other"],
    "sex": ["sex_Male"],
}
PAPER_COMPAS_FEATURE_TYPE = {
    "age": "numerical",
    "two_year_recid": "numerical",
    "priors_count": "numerical",
    "length_of_stay": "numerical",
    "c_charge_degree_M": "binary",
    "race_Other": "binary",
    "sex_Male": "binary",
}
PAPER_COMPAS_FEATURE_MUTABILITY = {
    "age": False,
    "two_year_recid": True,
    "priors_count": True,
    "length_of_stay": True,
    "c_charge_degree_M": True,
    "race_Other": False,
    "sex_Male": False,
}
PAPER_COMPAS_FEATURE_ACTIONABILITY = {
    "age": "none",
    "two_year_recid": "any",
    "priors_count": "any",
    "length_of_stay": "any",
    "c_charge_degree_M": "any",
    "race_Other": "none",
    "sex_Male": "none",
}


@dataclass(frozen=True)
class VariantConfig:
    name: str
    budget: int
    iterations: int
    selection: str
    candidate_count: int | None = None


@dataclass(frozen=True)
class CarlaDataSpec:
    target: str
    continuous: list[str]
    categorical: list[str]
    immutables: list[str]


def _to_feature_dataframe(
    values: pd.DataFrame | np.ndarray | torch.Tensor,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.loc[:, list(feature_names)].copy(deep=True)

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    if array.ndim == 1:
        array = array.reshape(1, -1)
    return pd.DataFrame(array, columns=list(feature_names))


class CarlaAnnTorchNetwork(nn.Module):
    def __init__(self, input_layer: int, hidden_layers: list[int], num_of_classes: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_layer, hidden_layers[0]))
        for index in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[index], hidden_layers[index + 1]))
        self.layers.append(nn.Linear(hidden_layers[-1], num_of_classes))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = self.relu(x)
            else:
                x = self.softmax(x)
        return x


class CarlaAnnModel(ModelObject):
    def __init__(
        self,
        feature_names: Sequence[str],
        data_spec: CarlaDataSpec,
        seed: int | None = None,
        device: str = "cpu",
        epochs: int = 20,
        learning_rate: float = 0.002,
        batch_size: int = 256,
        layers: list[int] | None = None,
    ) -> None:
        self.feature_input_order = list(feature_names)
        self.data = data_spec
        self._seed = seed
        self._device = device
        self._epochs = int(epochs)
        self._learning_rate = float(learning_rate)
        self._batch_size = int(batch_size)
        self._layers = list(layers or [10, 5, 10])
        self._need_grad = True
        self._is_trained = False

    @property
    def raw_model(self) -> nn.Module:
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        return self._model

    def _prepare_tensor(
        self,
        values: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(values, pd.DataFrame):
            array = values.loc[:, self.feature_input_order].to_numpy(dtype="float32")
            tensor = torch.tensor(array, dtype=torch.float32, device=self._device)
        elif isinstance(values, torch.Tensor):
            tensor = values.to(self._device, dtype=torch.float32)
        else:
            array = np.asarray(values, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            tensor = torch.tensor(array, dtype=torch.float32, device=self._device)

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _label_tensor(self, target_df: pd.DataFrame) -> torch.Tensor:
        target_series = target_df.iloc[:, 0].tolist()
        return torch.tensor(
            [self._class_to_index[value] for value in target_series],
            dtype=torch.long,
        )

    def fit(
        self,
        trainset,
        testset=None,
    ):
        if trainset is None:
            raise ValueError("trainset is required for CarlaAnnModel.fit()")
        if testset is None:
            raise ValueError("testset is required for CarlaAnnModel.fit()")

        with seed_context(self._seed):
            X_train, labels_train, output_dim = self.extract_training_data(trainset)
            if output_dim != 2:
                raise ValueError("CARLA ANN reproduction expects binary classification")

            X_test = testset.get(target=False).loc[:, self.feature_input_order].copy(deep=True)
            labels_test = self._label_tensor(testset.get(target=True))

            self._model = CarlaAnnTorchNetwork(
                input_layer=X_train.shape[1],
                hidden_layers=self._layers,
                num_of_classes=output_dim,
            ).to(self._device)

            train_dataset = TensorDataset(
                torch.tensor(
                    X_train.to_numpy(dtype="float32"),
                    dtype=torch.float32,
                ),
                labels_train.clone().detach().to(dtype=torch.long),
            )
            test_dataset = TensorDataset(
                torch.tensor(
                    X_test.to_numpy(dtype="float32"),
                    dtype=torch.float32,
                ),
                labels_test,
            )
            loaders = {
                "train": DataLoader(
                    train_dataset,
                    batch_size=self._batch_size,
                    shuffle=True,
                ),
                "test": DataLoader(
                    test_dataset,
                    batch_size=self._batch_size,
                    shuffle=True,
                ),
            }

            criterion = nn.BCELoss()
            optimizer = optim.RMSprop(self._model.parameters(), lr=self._learning_rate)

            epoch_iterator = tqdm(range(self._epochs), desc="carla-ann-fit", leave=False)
            for _ in epoch_iterator:
                for phase in ["train", "test"]:
                    if phase == "train":
                        self._model.train()
                    else:
                        self._model.eval()

                    for inputs, labels in loaders[phase]:
                        inputs = inputs.to(self._device)
                        labels = labels.to(self._device).to(dtype=torch.int64)
                        labels = torch.nn.functional.one_hot(labels, num_classes=2)

                        optimizer.zero_grad()
                        with torch.set_grad_enabled(phase == "train"):
                            outputs = self._model(inputs.float())
                            loss = criterion(outputs, labels.float())
                            if phase == "train":
                                loss.backward()
                                optimizer.step()

            self._model.eval()
            self._is_trained = True

    def predict_proba_features(
        self,
        values: pd.DataFrame | np.ndarray | torch.Tensor,
        preserve_grad: bool = False,
    ) -> np.ndarray | torch.Tensor:
        tensor = self._prepare_tensor(values)
        if preserve_grad:
            return self._model(tensor)
        with torch.no_grad():
            probabilities = self._model(tensor)
        return probabilities.detach().cpu().numpy()

    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        tensor = self._prepare_tensor(X)
        with torch.no_grad():
            probabilities = self._model(tensor)
        probabilities = probabilities.detach().cpu()
        if proba:
            return probabilities
        return probabilities.argmax(dim=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        return self._model(self._prepare_tensor(X))


class CarlaModelAdapter:
    def __init__(self, target_model: CarlaAnnModel, feature_names: Sequence[str]):
        self._target_model = target_model
        self._feature_names = list(feature_names)
        self.supports_gradients = True

    def get_ordered_features(
        self,
        X: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> pd.DataFrame:
        return _to_feature_dataframe(X, self._feature_names)

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        if isinstance(X, torch.Tensor):
            return self._target_model.predict_proba_features(X, preserve_grad=True)
        return self._target_model.predict_proba_features(
            self.get_ordered_features(X),
            preserve_grad=False,
        )

    def predict_label_indices(
        self,
        X: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        probabilities = self.predict_proba(X)
        if isinstance(probabilities, torch.Tensor):
            return probabilities.argmax(dim=1).detach().cpu().numpy()
        return np.asarray(probabilities, dtype=np.float32).argmax(axis=1)


def _flatten_paper_categorical_columns(feature_names: Sequence[str]) -> list[str]:
    ordered = [
        column
        for columns in PAPER_COMPAS_ENCODING.values()
        for column in columns
        if column in feature_names
    ]
    return ordered


def _carla_check_counterfactuals(
    model: CarlaAnnModel,
    counterfactuals: list[np.ndarray] | pd.DataFrame,
    factuals_index: pd.Index,
    desired_index: int,
) -> pd.DataFrame:
    if isinstance(counterfactuals, list):
        df_counterfactuals = pd.DataFrame(
            np.array(counterfactuals),
            columns=model.feature_input_order,
            index=factuals_index.copy(),
        )
    else:
        df_counterfactuals = counterfactuals.copy(deep=True)

    predictions = np.argmax(
        model.predict_proba_features(df_counterfactuals, preserve_grad=False),
        axis=1,
    )
    df_counterfactuals.loc[predictions != int(desired_index), :] = np.nan
    return df_counterfactuals.loc[:, model.feature_input_order].copy(deep=True)


def _carla_reconstruct_encoding_constraints(
    x: torch.Tensor,
    feature_pos: list[int],
    binary_cat_features: bool,
) -> torch.Tensor:
    x_encoded = x.clone()
    if binary_cat_features:
        for position in feature_pos:
            x_encoded[:, position] = torch.round(x_encoded[:, position])
        return x_encoded

    binary_pairs = list(zip(feature_pos[:-1], feature_pos[1:]))[0::2]
    for left_index, right_index in binary_pairs:
        temp = (x_encoded[:, left_index] >= x_encoded[:, right_index]).float()
        x_encoded[:, right_index] = (
            x_encoded[:, left_index] < x_encoded[:, right_index]
        ).float()
        x_encoded[:, left_index] = temp
    return x_encoded


def _carla_wachter_recourse(
    torch_model: nn.Module,
    x: np.ndarray,
    cat_feature_indices: list[int],
    immutables: list[int],
    binary_cat_features: bool,
    feature_costs,
    lr: float,
    lambda_param: float,
    y_target: list[int],
    n_iter: int,
    t_max_min: float,
    cost_fn,
    clamp: bool,
    loss_type: str,
    device: str,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    torch.manual_seed(0)

    if feature_costs is not None:
        feature_costs = torch.from_numpy(feature_costs).float().to(device)

    x_tensor = torch.from_numpy(x).float().to(device)
    y_target_tensor = torch.tensor(y_target).float().to(device)
    lamb = torch.tensor(lambda_param).float().to(device)
    x_new = torch.autograd.Variable(x_tensor.clone(), requires_grad=True)
    x_new_encoded = _carla_reconstruct_encoding_constraints(
        x_new,
        cat_feature_indices,
        binary_cat_features,
    )

    optimizer = optim.Adam([x_new], lr, amsgrad=True)

    if loss_type == "MSE":
        target_class = int(y_target_tensor[0] > 0.0)
        loss_fn = nn.MSELoss()
    elif loss_type == "BCE":
        target_class = torch.round(y_target_tensor[1]).int()
        loss_fn = nn.BCELoss()
    else:
        raise ValueError(f"loss_type {loss_type} not supported")

    f_x_new = torch_model(x_new)[:, target_class]
    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=t_max_min)
    total_iterations = 0
    lambda_reductions = 0

    while f_x_new <= DECISION_THRESHOLD:
        iteration = 0
        while f_x_new <= DECISION_THRESHOLD and iteration < int(n_iter):
            optimizer.zero_grad()
            x_new_encoded = _carla_reconstruct_encoding_constraints(
                x_new,
                cat_feature_indices,
                binary_cat_features,
            )
            f_x_new = torch_model(x_new_encoded)[:, target_class]

            if loss_type == "MSE":
                f_x_loss = torch.log(f_x_new / (1 - f_x_new))
            else:
                f_x_loss = torch_model(x_new_encoded).squeeze(axis=0)

            loss = loss_fn(f_x_loss, y_target_tensor) + lamb * cost_fn(x_new_encoded, x_tensor)
            loss.backward()
            if immutables:
                x_new.grad[:, immutables] = 0.0
            optimizer.step()
            if clamp:
                with torch.no_grad():
                    x_new.clamp_(0, 1)
            iteration += 1
            total_iterations += 1

        if f_x_new <= DECISION_THRESHOLD:
            lambda_reductions += 1
        lamb = lamb / 2
        lamb.clamp_(min=0)

        if datetime.datetime.now() - t0 > t_max:
            break
        if f_x_new >= DECISION_THRESHOLD:
            break

    x_new_encoded = _carla_reconstruct_encoding_constraints(
        x_new,
        cat_feature_indices,
        binary_cat_features,
    )
    final_probability = float(torch_model(x_new_encoded)[:, target_class].detach().cpu().item())
    diagnostics = {
        "iterations": float(total_iterations),
        "lambda_reductions": float(lambda_reductions),
        "target_probability": final_probability,
        "success": bool(final_probability > DECISION_THRESHOLD),
    }
    return x_new_encoded.detach().cpu().numpy().squeeze(axis=0), diagnostics


class CarlaWachterBaseline:
    def __init__(
        self,
        target_model: CarlaAnnModel,
        schema: FeatureSchema,
        desired_class: int | str,
        lr: float,
        lambda_param: float,
        n_iter: int,
        t_max_min: float,
        norm: int,
        loss_type: str,
    ) -> None:
        self._target_model = target_model
        self._schema = schema
        self._desired_class = desired_class
        self._feature_names = list(schema.feature_names)
        self._desired_index = int(target_model.get_class_to_index()[desired_class])
        self._categorical_indices = [
            self._feature_names.index(column)
            for column in _flatten_paper_categorical_columns(self._feature_names)
        ]
        self._immutable_indices = list(schema.immutable_indices)
        self._lr = float(lr)
        self._lambda_param = float(lambda_param)
        self._n_iter = int(n_iter)
        self._t_max_min = float(t_max_min)
        self._loss_type = str(loss_type).upper()
        self._row_diagnostics: list[dict[str, float | bool]] = []
        if self._desired_index == 1:
            self._y_target = [0, 1]
        else:
            self._y_target = [1, 0]
        self._cost_fn = lambda x, y: torch.linalg.norm(x - y, ord=int(norm))

    def clear_diagnostics(self) -> None:
        self._row_diagnostics = []

    def diagnostics(self) -> dict[str, float]:
        if not self._row_diagnostics:
            return {}
        iterations = np.array(
            [float(item["iterations"]) for item in self._row_diagnostics],
            dtype=np.float64,
        )
        lambda_reductions = np.array(
            [float(item["lambda_reductions"]) for item in self._row_diagnostics],
            dtype=np.float64,
        )
        target_probabilities = np.array(
            [float(item["target_probability"]) for item in self._row_diagnostics],
            dtype=np.float64,
        )
        successes = np.array(
            [float(item["success"]) for item in self._row_diagnostics],
            dtype=np.float64,
        )
        return {
            "base_iterations_mean": float(iterations.mean()),
            "base_lambda_reduced_fraction": float((lambda_reductions > 0.0).mean()),
            "base_target_probability_mean": float(target_probabilities.mean()),
            "base_target_probability_min": float(target_probabilities.min()),
            "base_target_probability_max": float(target_probabilities.max()),
            "base_internal_success_fraction": float(successes.mean()),
        }

    def generate(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        rows: list[np.ndarray] = []
        for _, row in factuals.iterrows():
            counterfactual, diagnostics = _carla_wachter_recourse(
                torch_model=self._target_model.raw_model,
                x=row.to_numpy(dtype=np.float32).reshape(1, -1),
                cat_feature_indices=self._categorical_indices,
                immutables=self._immutable_indices,
                binary_cat_features=True,
                feature_costs=None,
                lr=self._lr,
                lambda_param=self._lambda_param,
                y_target=self._y_target,
                n_iter=self._n_iter,
                t_max_min=self._t_max_min,
                cost_fn=self._cost_fn,
                clamp=True,
                loss_type=self._loss_type,
                device=self._target_model._device,
            )
            rows.append(counterfactual)
            self._row_diagnostics.append(diagnostics)

        return _carla_check_counterfactuals(
            model=self._target_model,
            counterfactuals=pd.DataFrame(rows, index=factuals.index, columns=self._feature_names),
            factuals_index=factuals.index,
            desired_index=self._desired_index,
        )


def _carla_hyper_sphere_coordinates(
    n_search_samples: int,
    instance: np.ndarray,
    high: float,
    low: float,
    seed: int,
    p_norm: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    with seed_context(seed):
        delta_instance = np.random.randn(n_search_samples, instance.shape[1])
        dist = np.random.rand(n_search_samples) * (high - low) + low
    norm_p = np.linalg.norm(delta_instance, ord=p_norm, axis=1)
    d_norm = np.divide(dist, norm_p, out=np.ones_like(dist), where=norm_p != 0).reshape(-1, 1)
    delta_instance = np.multiply(delta_instance, d_norm)
    candidate_counterfactuals = instance + delta_instance
    return candidate_counterfactuals, dist


def _carla_growing_spheres_search(
    instance: pd.DataFrame,
    keys_mutable: list[str],
    keys_immutable: list[str],
    continuous_cols: list[str],
    binary_cols: list[str],
    feature_order: list[str],
    model: CarlaAnnModel,
    seed: int,
    n_search_samples: int = 1000,
    p_norm: int = 1,
    step: float = 0.2,
    max_iter: int = 1000,
) -> np.ndarray:
    with seed_context(seed):
        keys_mutable_continuous = sorted(
            list(set(keys_mutable) - set(binary_cols)),
            key=feature_order.index,
        )
        keys_mutable_binary = sorted(
            list(set(keys_mutable) - set(continuous_cols)),
            key=feature_order.index,
        )

        instance_immutable_replicated = np.repeat(
            instance[keys_immutable].values.reshape(1, -1),
            n_search_samples,
            axis=0,
        )
        instance_replicated = np.repeat(
            instance.values.reshape(1, -1),
            n_search_samples,
            axis=0,
        )
        instance_mutable_replicated_continuous = np.repeat(
            instance[keys_mutable_continuous].values.reshape(1, -1),
            n_search_samples,
            axis=0,
        )
        instance_mutable_replicated_binary = np.repeat(
            instance[keys_mutable_binary].values.reshape(1, -1),
            n_search_samples,
            axis=0,
        )

        low = 0.0
        high = low + float(step)
        count = 0
        candidate_counterfactual_star = np.empty(instance_replicated.shape[1], dtype=np.float32)
        candidate_counterfactual_star[:] = np.nan
        instance_label = int(
            np.argmax(model.predict_proba_features(instance.values.reshape(1, -1), preserve_grad=False))
        )

        while count < int(max_iter):
            count += 1
            candidate_counterfactuals_continuous, _ = _carla_hyper_sphere_coordinates(
                n_search_samples=n_search_samples,
                instance=instance_mutable_replicated_continuous,
                high=high,
                low=low,
                seed=seed,
                p_norm=p_norm,
            )
            candidate_counterfactuals = pd.DataFrame(
                np.c_[
                    instance_immutable_replicated,
                    candidate_counterfactuals_continuous,
                    instance_mutable_replicated_binary,
                ],
                columns=keys_immutable + keys_mutable_continuous + keys_mutable_binary,
            )
            candidate_counterfactuals = candidate_counterfactuals[feature_order]

            if p_norm == 1:
                distances = np.abs(candidate_counterfactuals.values - instance_replicated).sum(axis=1)
            elif p_norm == 2:
                distances = np.square(candidate_counterfactuals.values - instance_replicated).sum(axis=1)
            else:
                raise ValueError("Distance not defined yet")

            y_candidate = np.argmax(
                model.predict_proba_features(candidate_counterfactuals.values, preserve_grad=False),
                axis=1,
            )
            indices = np.where(y_candidate != instance_label)
            candidate_counterfactuals_valid = candidate_counterfactuals.values[indices]
            candidate_distances = distances[indices]
            if len(candidate_distances) > 0:
                min_index = int(np.argmin(candidate_distances))
                candidate_counterfactual_star = candidate_counterfactuals_valid[min_index]
                break

            low = high
            high = low + float(step)

    return candidate_counterfactual_star


class CarlaGrowingSpheresGenerator:
    def __init__(
        self,
        target_model: CarlaAnnModel,
        schema: FeatureSchema,
        desired_class: int | str,
        n_search_samples: int,
        p_norm: int,
        step: float,
        max_iter: int,
    ) -> None:
        self._target_model = target_model
        self._schema = schema
        self._desired_class = desired_class
        self._desired_index = int(target_model.get_class_to_index()[desired_class])
        self._feature_names = list(schema.feature_names)
        self._continuous_feature_names = [
            self._feature_names[index] for index in schema.continuous_indices
        ]
        self._binary_feature_names = _flatten_paper_categorical_columns(self._feature_names)
        self._immutable_feature_names = [
            self._feature_names[index] for index in schema.immutable_indices
        ]
        self._mutable_feature_names = [
            feature_name
            for feature_name in self._feature_names
            if feature_name not in self._immutable_feature_names
        ]
        self._n_search_samples = int(n_search_samples)
        self._p_norm = int(p_norm)
        self._step = float(step)
        self._max_iter = int(max_iter)

    def generate(self, factuals: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        rows: list[np.ndarray] = []
        for row_index, (_, row) in enumerate(factuals.iterrows()):
            row_seed = int(seed) + row_index if seed is not None else 0
            rows.append(
                _carla_growing_spheres_search(
                    instance=row.to_frame().T.loc[:, self._feature_names],
                    keys_mutable=self._mutable_feature_names,
                    keys_immutable=self._immutable_feature_names,
                    continuous_cols=self._continuous_feature_names,
                    binary_cols=self._binary_feature_names,
                    feature_order=self._feature_names,
                    model=self._target_model,
                    seed=row_seed,
                    n_search_samples=self._n_search_samples,
                    p_norm=self._p_norm,
                    step=self._step,
                    max_iter=self._max_iter,
                )
            )

        return _carla_check_counterfactuals(
            model=self._target_model,
            counterfactuals=pd.DataFrame(rows, index=factuals.index, columns=self._feature_names),
            factuals_index=factuals.index,
            desired_index=self._desired_index,
        )


def _carla_sample_preference_weights(dimension: int) -> np.ndarray:
    centers = np.array([0.2, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    means = np.random.choice(centers, size=int(dimension))
    preference = np.random.multivariate_normal(means, cov=np.eye(int(dimension)))
    return np.clip(preference.astype(np.float32), a_min=0.1, a_max=5.0)


class CarlaSimulatedUser:
    def __init__(
        self,
        factual: pd.DataFrame,
        schema: FeatureSchema,
        gt_generator: CarlaGrowingSpheresGenerator,
        ground_truth_scale: str = "far",
        use_preferences: bool = False,
        noise_prob: float = 0.0,
        max_attempts: int = 64,
        seed: int | None = None,
    ) -> None:
        self._factual = factual.loc[:, schema.feature_names].copy(deep=True)
        self._schema = schema
        self._eps = {"near": 0.2, "intermediate": 0.8, "far": 1.5}[str(ground_truth_scale).lower()]
        self._use_preferences = bool(use_preferences)
        self._noise_prob = float(noise_prob)
        self._seed = seed
        self._ground_truth = self._generate_ground_truth_counterfactual(
            gt_generator=gt_generator,
            max_attempts=max_attempts,
        )
        self._preferences = _carla_sample_preference_weights(len(schema.feature_names))

    @property
    def factual(self) -> pd.DataFrame:
        return self._factual.copy(deep=True)

    @property
    def ground_truth(self) -> pd.DataFrame:
        return self._ground_truth.copy(deep=True)

    @property
    def preferences(self) -> np.ndarray:
        return self._preferences.copy()

    def _generate_ground_truth_counterfactual(
        self,
        gt_generator: CarlaGrowingSpheresGenerator,
        max_attempts: int,
    ) -> pd.DataFrame:
        factual_values = self._factual.to_numpy(dtype="float32").reshape(-1)
        mutable_continuous = list(self._schema.mutable_continuous_indices)
        count = 0

        while count < max(1, int(max_attempts)):
            perturbed = factual_values.copy()
            if mutable_continuous:
                direction = 2.0 * np.random.rand(len(mutable_continuous)) - 1.0
                norm = np.linalg.norm(direction, ord=2)
                if norm > 0.0:
                    direction = direction / norm * self._eps
                    perturbed[mutable_continuous] += direction.astype(np.float32)
            perturbed.clip(0.0, 1.0, out=perturbed)
            perturbed_df = pd.DataFrame(
                perturbed.reshape(1, -1),
                columns=self._schema.feature_names,
            )
            candidate = gt_generator.generate(
                perturbed_df,
                seed=None if self._seed is None else int(self._seed) + count,
            )
            if not candidate.isna().any(axis=1).iloc[0]:
                candidate = candidate.clip(lower=0.0, upper=1.0)
                return candidate.reset_index(drop=True)
            count += 1

        fallback = gt_generator.generate(
            self._factual,
            seed=None if self._seed is None else int(self._seed) + count,
        )
        if not fallback.isna().any(axis=1).iloc[0]:
            fallback = fallback.clip(lower=0.0, upper=1.0)
            return fallback.reset_index(drop=True)
        return self._factual.reset_index(drop=True).copy(deep=True)

    def compare(self, candidate_a: pd.DataFrame, candidate_b: pd.DataFrame) -> int:
        if not self._use_preferences:
            distance_a = np.linalg.norm(
                candidate_a.to_numpy(dtype="float32").reshape(-1)
                - self._ground_truth.to_numpy(dtype="float32").reshape(-1),
                ord=2,
            )
            distance_b = np.linalg.norm(
                candidate_b.to_numpy(dtype="float32").reshape(-1)
                - self._ground_truth.to_numpy(dtype="float32").reshape(-1),
                ord=2,
            )
        else:
            factual_values = self._factual.to_numpy(dtype="float32").reshape(-1)
            distance_a = np.dot(
                self._preferences,
                np.abs(candidate_a.to_numpy(dtype="float32").reshape(-1) - factual_values),
            )
            distance_b = np.dot(
                self._preferences,
                np.abs(candidate_b.to_numpy(dtype="float32").reshape(-1) - factual_values),
            )

        prefer_second = not (distance_a < distance_b)
        if self._noise_prob > 0.0:
            if np.random.choice([0, 1], p=[1.0 - self._noise_prob, self._noise_prob]):
                prefer_second = not prefer_second
        elif self._noise_prob < 0.0:
            distance = np.linalg.norm(
                candidate_a.to_numpy(dtype="float32").reshape(-1)
                - candidate_b.to_numpy(dtype="float32").reshape(-1),
                ord=2,
            )

            def _sigmoid(alpha: float, beta: float, x_value: float) -> float:
                return 1.0 / (1.0 + np.exp(-alpha * (x_value + beta)))

            flip_probability = 1.0 - _sigmoid(2.0, 1.0, float(distance))
            if np.random.choice([0, 1], p=[1.0 - flip_probability, flip_probability]):
                prefer_second = not prefer_second

        return int(prefer_second)


def _apply_carla_reference_defaults(config: dict) -> dict:
    cfg = deepcopy(config)
    cfg.setdefault("model", {})
    cfg.setdefault("method", {})
    cfg["model"].update(CARLA_REFERENCE_MODEL_DEFAULTS)
    cfg["method"].update(CARLA_REFERENCE_WACHTER_DEFAULTS)
    return cfg


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must parse to a dictionary")
    return config


def _apply_device(config: dict, device: str) -> dict:
    cfg = deepcopy(config)
    cfg.setdefault("model", {})
    cfg.setdefault("method", {})
    cfg["model"]["device"] = device
    cfg["method"]["device"] = device
    return cfg


def _apply_seed(config: dict, seed: int) -> dict:
    cfg = deepcopy(config)
    for preprocess_cfg in cfg.get("preprocess", []):
        if isinstance(preprocess_cfg, dict) and "seed" in preprocess_cfg:
            preprocess_cfg["seed"] = int(seed)
    if "model" in cfg and isinstance(cfg["model"], dict):
        cfg["model"]["seed"] = int(seed)
    if "method" in cfg and isinstance(cfg["method"], dict):
        cfg["method"]["seed"] = int(seed)
    logger_cfg = cfg.get("logger")
    if isinstance(logger_cfg, dict) and logger_cfg.get("path"):
        log_path = Path(str(logger_cfg["path"]))
        logger_cfg["path"] = str(
            log_path.with_name(f"{log_path.stem}_seed_{seed}{log_path.suffix}")
        )
    cfg["name"] = f"{cfg.get('name', 'hare_reproduce')}_seed_{seed}"
    return cfg


def _reference_style_split(dataset, split_preprocess) -> tuple[object, object]:
    df = dataset.snapshot()
    split = split_preprocess._split
    sample = split_preprocess._sample
    seed = split_preprocess._seed
    stratify = None
    target_column = dataset.target_column
    if target_column in df.columns:
        target_series = df[target_column]
        if target_series.nunique(dropna=True) > 1:
            stratify = target_series

    if isinstance(split, float):
        train_df, test_df = train_test_split(
            df,
            train_size=1.0 - split,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=split,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )

    if sample is not None:
        test_df = test_df.sample(n=sample, random_state=seed).copy(deep=True)
    else:
        test_df = test_df.copy(deep=True)
    train_df = train_df.copy(deep=True)

    trainset = dataset
    testset = dataset.clone()
    trainset.update("trainset", True, df=train_df)
    testset.update("testset", True, df=test_df)
    return trainset, testset


def _reference_compas_split_cache_dir() -> Path:
    return PROJECT_ROOT / "cache" / "hare_reference_data"


def _load_reference_compas_raw_split(split_name: str) -> pd.DataFrame:
    if split_name not in REFERENCE_COMPAS_SPLIT_URLS:
        raise ValueError(f"Unsupported split name: {split_name}")

    cache_dir = _reference_compas_split_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"compas_{split_name}.csv"
    if not cache_path.exists():
        try:
            urlretrieve(REFERENCE_COMPAS_SPLIT_URLS[split_name], cache_path)
        except Exception as error:
            raise RuntimeError(
                "Reference COMPAS split is missing and could not be downloaded. "
                "Populate cache/hare_reference_data/compas_train.csv and "
                "cache/hare_reference_data/compas_test.csv before rerunning."
            ) from error
    return pd.read_csv(cache_path)


def _resolve_reference_split_indices(
    raw_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> list[int]:
    source = raw_df.reset_index().rename(columns={"index": "__source_index"})
    split = split_df.copy(deep=True)
    group_columns = list(raw_df.columns)
    source["__hare_occurrence"] = source.groupby(group_columns, dropna=False).cumcount()
    split["__hare_occurrence"] = split.groupby(group_columns, dropna=False).cumcount()
    merged = split.merge(
        source,
        on=group_columns + ["__hare_occurrence"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged["__source_index"].isna().any():
        raise ValueError("Failed to map the reference COMPAS split onto the local dataset")
    return merged["__source_index"].astype(int).tolist()


def _resolve_reference_compas_indices(dataset) -> tuple[list[int], list[int]] | None:
    if getattr(dataset, "name", None) != "compas_carla":
        return None

    raw_df = dataset.snapshot().reset_index(drop=True)
    train_raw = _load_reference_compas_raw_split("train")
    test_raw = _load_reference_compas_raw_split("test")
    train_indices = _resolve_reference_split_indices(raw_df, train_raw)
    test_indices = _resolve_reference_split_indices(raw_df, test_raw)
    return train_indices, test_indices


def _reference_compas_split(
    dataset,
    split_preprocess,
    train_indices: list[int],
    test_indices: list[int],
) -> tuple[object, object]:
    df = dataset.snapshot()
    seed = split_preprocess._seed
    sample = split_preprocess._sample

    train_df = df.iloc[train_indices].copy(deep=True)
    test_df = df.iloc[test_indices].copy(deep=True)
    if sample is not None:
        test_df = test_df.sample(n=sample, random_state=seed).copy(deep=True)

    trainset = dataset
    testset = dataset.clone()
    trainset.update("trainset", True, df=train_df)
    testset.update("testset", True, df=test_df)
    return trainset, testset


def _should_align_paper_compas(dataset) -> bool:
    return (
        getattr(dataset, "name", None) == "compas_carla"
        and getattr(dataset, "target_column", None) == PAPER_COMPAS_TARGET
        and not getattr(dataset, "paper_compas_aligned", False)
    )


def _align_paper_compas_dataset(dataset):
    if not _should_align_paper_compas(dataset):
        return dataset
    if not hasattr(dataset, "encoding"):
        return dataset

    df = dataset.snapshot()
    missing_columns = [
        source_name
        for source_name in PAPER_COMPAS_SOURCE_COLUMNS.values()
        if source_name not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Paper COMPAS alignment is missing encoded columns: "
            f"{missing_columns}"
        )

    aligned_df = df.loc[:, list(PAPER_COMPAS_SOURCE_COLUMNS.values())].copy(deep=True)
    aligned_df.columns = list(PAPER_COMPAS_SOURCE_COLUMNS.keys())
    aligned_df = aligned_df.loc[:, PAPER_COMPAS_DF_ORDER].copy(deep=True)

    dataset.update("paper_compas_aligned", True, df=aligned_df)
    dataset.update("reordered", list(PAPER_COMPAS_FEATURE_ORDER))
    dataset.update("encoding", deepcopy(PAPER_COMPAS_ENCODING))
    dataset.update("encoded_feature_type", deepcopy(PAPER_COMPAS_FEATURE_TYPE))
    dataset.update(
        "encoded_feature_mutability",
        deepcopy(PAPER_COMPAS_FEATURE_MUTABILITY),
    )
    dataset.update(
        "encoded_feature_actionability",
        deepcopy(PAPER_COMPAS_FEATURE_ACTIONABILITY),
    )
    return dataset


def _materialize_datasets(experiment: Experiment) -> tuple[object, object]:
    reference_compas_indices = _resolve_reference_compas_indices(experiment._raw_dataset)
    datasets = [experiment._raw_dataset]
    for preprocess_step in experiment._preprocess:
        next_datasets = []
        for current_dataset in datasets:
            if preprocess_step.__class__.__name__ == "SplitPreProcess":
                if reference_compas_indices is not None:
                    transformed = _reference_compas_split(
                        current_dataset,
                        preprocess_step,
                        train_indices=reference_compas_indices[0],
                        test_indices=reference_compas_indices[1],
                    )
                else:
                    transformed = _reference_style_split(current_dataset, preprocess_step)
            else:
                transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                transformed = _align_paper_compas_dataset(transformed)
                next_datasets.append(transformed)
        datasets = next_datasets
    return experiment._resolve_train_test(datasets)


def _resolve_round_query_budgets(budget: int, iterations: int) -> list[int]:
    active_rounds = min(int(iterations), int(budget))
    if active_rounds < 1:
        raise ValueError("budget and iterations must allow at least one active round")
    base_budget = int(budget) // active_rounds
    remainder = int(budget) % active_rounds
    return [
        base_budget + (1 if round_index < remainder else 0)
        for round_index in range(active_rounds)
    ]


def _resolve_reference_multi_round_query_budgets(
    budget: int,
    candidate_count: int,
) -> list[int]:
    if int(candidate_count) < 1:
        raise ValueError("candidate_count must be >= 1")

    full_rounds, remainder = divmod(int(budget), int(candidate_count))
    budgets = [int(candidate_count) for _ in range(full_rounds)]
    if remainder:
        budgets.append(int(remainder))
    if not budgets:
        budgets.append(int(candidate_count))
    return budgets


def _select_negative_factuals(
    model,
    testset,
    desired_class: int | str,
    num_instances: int,
) -> pd.DataFrame:
    desired_index = int(model.get_class_to_index()[desired_class])
    probabilities = model.predict_proba(testset).detach().cpu().numpy()
    predicted_indices = probabilities.argmax(axis=1)
    feature_df = testset.get(target=False).reset_index(drop=True)
    factuals = feature_df.loc[predicted_indices != desired_index].copy(deep=True)
    if factuals.empty:
        raise RuntimeError("No negatively predicted COMPAS factuals are available")
    return factuals.iloc[: int(num_instances)].reset_index(drop=True)


def _class_count_dict(values: np.ndarray) -> dict[int, int]:
    unique_values, counts = np.unique(values, return_counts=True)
    return {
        int(unique_value): int(count)
        for unique_value, count in zip(unique_values, counts, strict=False)
    }


def _rebalance_trainset(trainset, seed: int):
    target_column = trainset.target_column
    combined = pd.concat([trainset.get(target=False), trainset.get(target=True)], axis=1)
    target_series = combined[target_column]
    class_counts = target_series.value_counts(dropna=False)
    if class_counts.empty or class_counts.nunique() == 1:
        return trainset

    target_size = int(class_counts.max())
    balanced_parts: list[pd.DataFrame] = []
    for class_value in class_counts.index.tolist():
        class_df = combined.loc[target_series == class_value].copy(deep=True)
        replace = class_df.shape[0] < target_size
        balanced_parts.append(
            class_df.sample(
                n=target_size,
                replace=replace,
                random_state=int(seed),
            )
        )

    balanced_df = (
        pd.concat(balanced_parts, axis=0, ignore_index=True)
        .sample(frac=1.0, random_state=int(seed))
        .reset_index(drop=True)
    )
    balanced_trainset = trainset.clone()
    balanced_trainset.update("trainset", True, df=balanced_df)
    balanced_trainset.freeze()
    return balanced_trainset


def _evaluate_model_diagnostics(model, testset, desired_class: int | str) -> dict[str, object]:
    probabilities = model.predict_proba(testset).detach().cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    target = testset.get(target=True).iloc[:, 0].astype(int).to_numpy()
    desired_index = int(model.get_class_to_index()[desired_class])
    negative_mask = predictions != desired_index
    return {
        "test_accuracy": float((predictions == target).mean()),
        "pred_counts": _class_count_dict(predictions),
        "test_counts": _class_count_dict(target),
        "negatives_available": int(negative_mask.sum()),
    }


def _build_shared_hare_objects(
    model,
    trainset,
    method_cfg: dict,
) -> tuple[FeatureSchema, CarlaModelAdapter, object, object]:
    schema = resolve_feature_schema(trainset)
    adapter = CarlaModelAdapter(model, schema.feature_names)
    desired_class = method_cfg["desired_class"]
    if str(method_cfg.get("base_method", "wachter")).lower() != "wachter":
        raise ValueError("This reproduction script currently supports CARLA-parity Wachter only")

    baseline = CarlaWachterBaseline(
        target_model=model,
        schema=schema,
        desired_class=desired_class,
        lr=method_cfg.get("baseline_lr", CARLA_REFERENCE_WACHTER_DEFAULTS["baseline_lr"]),
        lambda_param=method_cfg.get(
            "baseline_lambda",
            CARLA_REFERENCE_WACHTER_DEFAULTS["baseline_lambda"],
        ),
        n_iter=method_cfg.get(
            "baseline_n_iter",
            CARLA_REFERENCE_WACHTER_DEFAULTS["baseline_n_iter"],
        ),
        t_max_min=method_cfg.get(
            "baseline_t_max_min",
            CARLA_REFERENCE_WACHTER_DEFAULTS["baseline_t_max_min"],
        ),
        norm=method_cfg.get("baseline_norm", CARLA_REFERENCE_WACHTER_DEFAULTS["baseline_norm"]),
        loss_type=method_cfg.get(
            "baseline_loss_type",
            CARLA_REFERENCE_WACHTER_DEFAULTS["baseline_loss_type"],
        ),
    )
    gt_generator = CarlaGrowingSpheresGenerator(
        target_model=model,
        schema=schema,
        desired_class=desired_class,
        n_search_samples=method_cfg.get("gt_n_search_samples", 300),
        step=method_cfg.get("gt_step", 0.2),
        max_iter=method_cfg.get("gt_max_iter", 1000),
        p_norm=1,
    )
    return schema, adapter, baseline, gt_generator


def _collect_metric_debug_rows(
    factuals: pd.DataFrame,
    ground_truths: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    limit: int,
) -> list[dict[str, object]]:
    debug_rows: list[dict[str, object]] = []
    factual_values = factuals.to_numpy(dtype="float32")
    ground_truth_values = ground_truths.to_numpy(dtype="float32")
    counterfactual_values = counterfactuals.to_numpy(dtype="float32")
    valid_mask = ~(
        np.isnan(ground_truth_values).any(axis=1) | np.isnan(counterfactual_values).any(axis=1)
    )
    valid_indices = np.where(valid_mask)[0][: max(0, int(limit))]
    for row_index in valid_indices.tolist():
        factual = factual_values[row_index]
        ground_truth = ground_truth_values[row_index]
        counterfactual = counterfactual_values[row_index]
        gt_delta = ground_truth - factual
        cf_delta = counterfactual - factual
        difference_mask = ~np.isclose(
            cf_delta,
            np.zeros_like(cf_delta),
            atol=1e-5,
            rtol=0.0,
        )
        debug_rows.append(
            {
                "row_index": int(row_index),
                "factual": factual.round(6).tolist(),
                "ground_truth": ground_truth.round(6).tolist(),
                "counterfactual": counterfactual.round(6).tolist(),
                "gt_delta": gt_delta.round(6).tolist(),
                "cf_delta": cf_delta.round(6).tolist(),
                "gtp": float(np.linalg.norm(ground_truth - counterfactual, ord=2)),
                "gtd": float(_safe_cosine_distance(gt_delta, cf_delta)),
                "proximity": float(np.square(np.abs(cf_delta)).sum()),
                "sparsity": float(difference_mask.sum()),
            }
        )
    return debug_rows


def _repeat_factual(factual: pd.DataFrame, times: int) -> pd.DataFrame:
    return pd.concat([factual] * int(times), axis=0, ignore_index=True)


def _build_user(
    factual: pd.DataFrame,
    schema: FeatureSchema,
    gt_generator,
    method_cfg: dict,
    seed: int | None = None,
) -> CarlaSimulatedUser:
    return CarlaSimulatedUser(
        factual=factual,
        schema=schema,
        gt_generator=gt_generator,
        ground_truth_scale=method_cfg.get("ground_truth_scale", "far"),
        use_preferences=bool(method_cfg.get("use_preferences", False)),
        noise_prob=float(method_cfg.get("noise_prob", 0.0)),
        max_attempts=int(method_cfg.get("gt_max_attempts", 64)),
        seed=seed,
    )


def _run_search_variant(
    factual: pd.DataFrame,
    user,
    initial_counterfactual: pd.DataFrame,
    adapter,
    target_model,
    schema: FeatureSchema,
    method_cfg: dict,
    round_query_budgets: list[int],
    selection: str,
    trace_steps: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    desired_class = method_cfg["desired_class"]
    target_index = int(target_model.get_class_to_index()[desired_class])
    current_counterfactual = initial_counterfactual.reset_index(drop=True).copy(deep=True)
    if current_counterfactual.isna().any(axis=1).iloc[0]:
        return current_counterfactual.reset_index(drop=True)

    calibrate_all_candidates = bool(method_cfg.get("calibrate_all_candidates", True))
    for round_budget in round_query_budgets:
        candidates = actionable_sampling(
            baseline=current_counterfactual,
            factual=factual,
            model=adapter,
            schema=schema,
            target_index=target_index,
            num_candidates=int(round_budget),
            radius=float(method_cfg.get("sampling_radius", 1.0)),
            lambda_=float(method_cfg.get("sampling_lambda", 10.0)),
            lr=float(method_cfg.get("sampling_lr", 0.1)),
            epochs=int(method_cfg.get("sampling_epochs", 100)),
        )
        sampled_candidates = candidates.copy(deep=True)

        if calibrate_all_candidates:
            candidates = calibrate_candidate_set(
                factual=factual,
                candidates=candidates,
                model=adapter,
                target_index=target_index,
                epsilon=float(method_cfg.get("boundary_epsilon", 1e-6)),
            )
            candidates = validate_counterfactuals(
                target_model=target_model,
                factuals=_repeat_factual(factual, candidates.shape[0]),
                candidates=candidates,
                desired_class=desired_class,
            )
            candidates = candidates.dropna(axis=0, how="any").reset_index(drop=True)

        if candidates.empty:
            break

        if selection == "random":
            winner_index = int(np.random.randint(0, candidates.shape[0]))
        elif selection == "oracle":
            winner_index = select_best_candidate(user, candidates)
        else:
            raise ValueError(f"Unsupported selection mode: {selection}")

        current_counterfactual = candidates.iloc[[winner_index]].copy(deep=True)
        if not calibrate_all_candidates:
            current_counterfactual = boundary_point_search(
                factual=factual,
                candidate=current_counterfactual,
                model=adapter,
                target_index=target_index,
                epsilon=float(method_cfg.get("boundary_epsilon", 1e-6)),
            )

        if trace_steps is not None:
            trace_steps.append(
                {
                    "round_budget": int(round_budget),
                    "selection": selection,
                    "sampled_candidates": _trace_dataframe(sampled_candidates),
                    "validated_candidates": _trace_dataframe(candidates),
                    "winner_index": int(winner_index),
                    "selected_counterfactual": _trace_dataframe(current_counterfactual),
                }
            )

    return validate_counterfactuals(
        target_model=target_model,
        factuals=factual.reset_index(drop=True),
        candidates=current_counterfactual.reset_index(drop=True),
        desired_class=desired_class,
    ).reset_index(drop=True)


def _safe_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a, ord=2))
    b_norm = float(np.linalg.norm(b, ord=2))
    if a_norm == 0.0 and b_norm == 0.0:
        return 0.0
    if a_norm == 0.0 or b_norm == 0.0:
        return 1.0
    return float(scipy_cosine(a, b))


def _compute_gt_metrics(
    factuals: pd.DataFrame,
    ground_truths: pd.DataFrame,
    counterfactuals: pd.DataFrame,
) -> dict[str, float]:
    gt_values = ground_truths.to_numpy(dtype="float32")
    cf_values = counterfactuals.to_numpy(dtype="float32")
    factual_values = factuals.to_numpy(dtype="float32")
    valid_mask = ~(np.isnan(gt_values).any(axis=1) | np.isnan(cf_values).any(axis=1))
    if not bool(valid_mask.any()):
        return {"gtp": float("nan"), "gtd": float("nan")}

    gt_values = gt_values[valid_mask]
    cf_values = cf_values[valid_mask]
    factual_values = factual_values[valid_mask]

    gtp = np.linalg.norm(gt_values - cf_values, ord=2, axis=1)
    gtd = np.array(
        [
            _safe_cosine_distance(
                gt_values[index] - factual_values[index],
                cf_values[index] - factual_values[index],
            )
            for index in range(gt_values.shape[0])
        ],
        dtype=np.float32,
    )
    return {"gtp": float(gtp.mean()), "gtd": float(gtd.mean())}


def _success_mask(counterfactuals: pd.DataFrame) -> np.ndarray:
    return ~counterfactuals.isna().any(axis=1).to_numpy()


def _compute_success_rate(counterfactuals: pd.DataFrame) -> float:
    mask = _success_mask(counterfactuals)
    if mask.size == 0:
        return float("nan")
    return float(mask.mean())


def _compute_distance_metrics(
    factuals: pd.DataFrame,
    counterfactuals: pd.DataFrame,
) -> dict[str, float]:
    mask = _success_mask(counterfactuals)
    if not bool(mask.any()):
        return {"proximity": float("nan"), "sparsity": float("nan")}

    delta = (
        counterfactuals.loc[mask].to_numpy(dtype="float32")
        - factuals.loc[mask].to_numpy(dtype="float32")
    )
    difference_mask = ~np.isclose(delta, np.zeros_like(delta), atol=1e-5)
    sparsity = np.sum(difference_mask, axis=1, dtype=float).mean()
    proximity = np.sum(np.square(np.abs(delta)), axis=1, dtype=float).mean()
    return {"proximity": float(proximity), "sparsity": float(sparsity)}


def _constraint_units(schema: FeatureSchema) -> tuple[list[list[int]], list[int]]:
    grouped_indices = {index for group in schema.categorical_groups for index in group}
    immutable_groups = [
        group for group in schema.categorical_groups if group[0] in schema.immutable_indices
    ]
    immutable_standalone = [
        index
        for index in schema.immutable_indices
        if index not in grouped_indices
    ]
    return immutable_groups, immutable_standalone


def _compute_constraint_violation(
    factuals: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    schema: FeatureSchema,
) -> float:
    mask = _success_mask(counterfactuals)
    if not bool(mask.any()):
        return float("nan")

    factual_values = factuals.loc[mask].to_numpy(dtype="float32")
    counterfactual_values = counterfactuals.loc[mask].to_numpy(dtype="float32")
    immutable_groups, immutable_standalone = _constraint_units(schema)
    violations: list[float] = []
    for factual, counterfactual in zip(factual_values, counterfactual_values, strict=False):
        count = 0
        for group in immutable_groups:
            if not np.allclose(counterfactual[group], factual[group], atol=1e-5, rtol=0.0):
                count += 1
        for index in immutable_standalone:
            if not np.isclose(counterfactual[index], factual[index], atol=1e-5, rtol=0.0):
                count += 1
        violations.append(float(count))
    return float(np.mean(violations))


def _compute_redundancy(
    factuals: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    adapter,
    desired_index: int,
) -> float:
    mask = _success_mask(counterfactuals)
    if not bool(mask.any()):
        return float("nan")

    factual_values = factuals.loc[mask].to_numpy(dtype="float32")
    counterfactual_values = counterfactuals.loc[mask].to_numpy(dtype="float32")
    redundancies: list[float] = []
    for factual, counterfactual in zip(factual_values, counterfactual_values, strict=False):
        redundancy = 0
        for column_index in range(counterfactual.shape[0]):
            if factual[column_index] == counterfactual[column_index]:
                continue
            temp_counterfactual = np.copy(counterfactual)
            temp_counterfactual[column_index] = factual[column_index]
            temp_prediction = int(adapter.predict_label_indices(temp_counterfactual.reshape(1, -1))[0])
            if temp_prediction == desired_index:
                redundancy += 1
        redundancies.append(float(redundancy))
    return float(np.mean(redundancies))


def _compute_variant_metrics(
    factuals: pd.DataFrame,
    ground_truths: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    schema: FeatureSchema,
    adapter,
    desired_index: int,
) -> dict[str, float]:
    metrics = {}
    metrics.update(_compute_gt_metrics(factuals, ground_truths, counterfactuals))
    metrics["success_rate"] = _compute_success_rate(counterfactuals)
    metrics["constraint_violation"] = _compute_constraint_violation(
        factuals,
        counterfactuals,
        schema,
    )
    metrics["redundancy"] = _compute_redundancy(
        factuals,
        counterfactuals,
        adapter,
        desired_index,
    )
    metrics.update(_compute_distance_metrics(factuals, counterfactuals))
    return metrics


def _metric_mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    if np.isnan(array).all():
        return float("nan"), float("nan")
    return float(np.nanmean(array)), float(np.nanstd(array))


def _format_mean_std(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "nan"
    if np.isnan(std):
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def _count_valid_rows(counterfactuals: pd.DataFrame) -> int:
    return int((~counterfactuals.isna().any(axis=1)).sum())


def _round_nested(values, decimals: int = 6):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return round(float(array), decimals)
    return np.round(array, decimals=decimals).tolist()


def _trace_counterfactual_metrics(
    factual: pd.DataFrame,
    ground_truth: pd.DataFrame,
    counterfactual: pd.DataFrame,
) -> dict[str, float | bool]:
    cf_valid = not counterfactual.isna().any(axis=1).iloc[0]
    gt_valid = not ground_truth.isna().any(axis=1).iloc[0]
    if not cf_valid or not gt_valid:
        return {
            "valid": False,
            "gtp": float("nan"),
            "gtd": float("nan"),
            "proximity": float("nan"),
            "sparsity": float("nan"),
        }

    factual_values = factual.to_numpy(dtype="float32").reshape(-1)
    gt_values = ground_truth.to_numpy(dtype="float32").reshape(-1)
    cf_values = counterfactual.to_numpy(dtype="float32").reshape(-1)
    gt_delta = gt_values - factual_values
    cf_delta = cf_values - factual_values
    difference_mask = ~np.isclose(cf_delta, np.zeros_like(cf_delta), atol=1e-5, rtol=0.0)
    return {
        "valid": True,
        "gtp": float(np.linalg.norm(gt_values - cf_values, ord=2)),
        "gtd": float(_safe_cosine_distance(gt_delta, cf_delta)),
        "proximity": float(np.square(np.abs(cf_delta)).sum()),
        "sparsity": float(difference_mask.sum()),
    }


def _trace_dataframe(df: pd.DataFrame | None) -> list[dict[str, float]] | None:
    if df is None:
        return None
    if df.empty:
        return []
    return [
        {column: round(float(value), 6) for column, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _aggregate_seed_metrics(seed_metrics: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    for variant_name, metrics_list in seed_metrics.items():
        metric_names = sorted({key for metrics in metrics_list for key in metrics})
        aggregates[variant_name] = {}
        for metric_name in metric_names:
            values = [float(metrics.get(metric_name, float("nan"))) for metrics in metrics_list]
            mean, std = _metric_mean_std(values)
            aggregates[variant_name][metric_name] = {"mean": mean, "std": std}
    return aggregates


def _table1_dataframe(
    aggregates: dict[str, dict[str, dict[str, float]]],
    compare_paper_targets: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name in ["base", "random", "hare", "multi_hare"]:
        if variant_name not in aggregates:
            continue
        row = {"Variant": VARIANT_LABELS[variant_name]}
        row["Reproduced GTP"] = _format_mean_std(
            aggregates[variant_name]["gtp"]["mean"],
            aggregates[variant_name]["gtp"]["std"],
        )
        row["Reproduced GTD"] = _format_mean_std(
            aggregates[variant_name]["gtd"]["mean"],
            aggregates[variant_name]["gtd"]["std"],
        )
        if compare_paper_targets:
            target = TABLE1_TARGETS[variant_name]
            row["Paper GTP"] = f'{target["gtp"]:.2f}'
            row["Paper GTD"] = f'{target["gtd"]:.2f}'
        rows.append(row)
    return pd.DataFrame(rows)


def _table4_dataframe(
    aggregates: dict[str, dict[str, dict[str, float]]],
    compare_paper_targets: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_map = [
        ("gtp", "GTP"),
        ("gtd", "GTD"),
        ("success_rate", "Succ"),
        ("constraint_violation", "ConVio"),
        ("redundancy", "Red"),
        ("proximity", "Pro"),
        ("sparsity", "Spa"),
    ]
    for variant_name in ["base", "hare", "multi_hare"]:
        if variant_name not in aggregates:
            continue
        row = {"Variant": VARIANT_LABELS[variant_name]}
        for metric_name, display_name in metric_map:
            row[f"Reproduced {display_name}"] = _format_mean_std(
                aggregates[variant_name][metric_name]["mean"],
                aggregates[variant_name][metric_name]["std"],
            )
            if compare_paper_targets:
                row[f"Paper {display_name}"] = f'{TABLE4_ANN_TARGETS[variant_name][metric_name]:.2f}'
        rows.append(row)
    return pd.DataFrame(rows)


def _variant_configs(
    budget: int,
    iterations: int,
    variant_scope: str,
) -> list[VariantConfig]:
    variant_scope = str(variant_scope).lower()
    if variant_scope in {"base", "trace_base"}:
        return [VariantConfig(name="base", budget=0, iterations=0, selection="baseline")]
    if variant_scope != "all":
        raise ValueError("variant_scope must be one of {'base', 'trace_base', 'all'}")
    return [
        VariantConfig(name="base", budget=0, iterations=0, selection="baseline"),
        VariantConfig(name="random", budget=int(budget), iterations=1, selection="random"),
        VariantConfig(name="hare", budget=int(budget), iterations=1, selection="oracle"),
        VariantConfig(
            name="multi_hare",
            budget=int(budget),
            iterations=max(1, int(np.ceil(int(budget) / int(iterations)))),
            selection="oracle",
            candidate_count=int(iterations),
        ),
    ]


def _run_seed(
    config: dict,
    seed: int,
    num_instances: int,
    budget: int,
    iterations: int,
    ground_truth_scale: str,
    variant_scope: str,
    train_balance: str,
    metric_debug: bool,
    metric_debug_limit: int,
    trace_seed: int | None,
    trace_limit: int,
) -> dict[str, object]:
    seeded_config = _apply_seed(config, seed)
    seeded_config["method"]["ground_truth_scale"] = ground_truth_scale
    seeded_config["method"]["budget"] = int(budget)
    seeded_config["method"]["iterations"] = 1
    print(
        f"\n[seed {seed}] building experiment and training "
        f"{seeded_config['model']['name']} on COMPAS"
    )
    experiment = Experiment(seeded_config)
    trainset, testset = _materialize_datasets(experiment)
    schema = resolve_feature_schema(trainset)
    desired_class = seeded_config["method"].get("desired_class", 1)
    data_spec = CarlaDataSpec(
        target=trainset.target_column,
        continuous=[schema.feature_names[index] for index in schema.continuous_indices],
        categorical=list(PAPER_COMPAS_ENCODING.keys()),
        immutables=[schema.feature_names[index] for index in schema.immutable_indices],
    )
    model_trainset = trainset
    train_balance_used = "off"
    if train_balance == "rebalance":
        model_trainset = _rebalance_trainset(trainset, seed)
        train_balance_used = "rebalance"
    elif train_balance not in {"off", "auto"}:
        raise ValueError("train_balance must be one of {'auto', 'rebalance', 'off'}")

    local_model = CarlaAnnModel(
        feature_names=schema.feature_names,
        data_spec=data_spec,
        seed=seeded_config["model"].get("seed"),
        device=seeded_config["model"].get("device", "cpu"),
        epochs=seeded_config["model"].get("epochs", CARLA_REFERENCE_MODEL_DEFAULTS["epochs"]),
        learning_rate=seeded_config["model"].get(
            "learning_rate",
            CARLA_REFERENCE_MODEL_DEFAULTS["learning_rate"],
        ),
        batch_size=seeded_config["model"].get(
            "batch_size",
            CARLA_REFERENCE_MODEL_DEFAULTS["batch_size"],
        ),
        layers=seeded_config["model"].get("layers", CARLA_REFERENCE_MODEL_DEFAULTS["layers"]),
    )
    local_model.fit(model_trainset, testset=testset)
    diagnostics = _evaluate_model_diagnostics(
        local_model,
        testset,
        desired_class=desired_class,
    )
    if train_balance == "auto" and diagnostics["negatives_available"] == 0:
        print(
            f"[seed {seed}] no negative factuals with unbalanced ANN; retrying with balanced trainset"
        )
        model_trainset = _rebalance_trainset(trainset, seed)
        train_balance_used = "rebalance"
        local_model.fit(model_trainset, testset=testset)
        diagnostics = _evaluate_model_diagnostics(
            local_model,
            testset,
            desired_class=desired_class,
        )

    train_target = model_trainset.get(target=True).iloc[:, 0].astype(int).to_numpy()
    print(
        f"[seed {seed}] train_balance={train_balance_used} "
        f"train_counts={_class_count_dict(train_target)} "
        f"test_counts={diagnostics['test_counts']} "
        f"pred_counts={diagnostics['pred_counts']} "
        f"test_acc={diagnostics['test_accuracy']:.4f} "
        f"negatives_available={diagnostics['negatives_available']}"
    )

    try:
        factuals = _select_negative_factuals(
            local_model,
            testset,
            desired_class=desired_class,
            num_instances=num_instances,
        )
    except RuntimeError as error:
        print(f"[seed {seed}] warning: {error}; skipping this seed")
        nan_metrics = {
            "gtp": float("nan"),
            "gtd": float("nan"),
            "success_rate": float("nan"),
            "constraint_violation": float("nan"),
            "redundancy": float("nan"),
            "proximity": float("nan"),
            "sparsity": float("nan"),
        }
        return {
            "seed": int(seed),
            "num_factuals": 0,
            "metrics": {
                variant_cfg.name: dict(nan_metrics)
                for variant_cfg in _variant_configs(budget, iterations, variant_scope)
            },
            "skipped_reason": str(error),
            "model_diagnostics": diagnostics,
        }
    schema, adapter, baseline, gt_generator = _build_shared_hare_objects(
        model=local_model,
        trainset=trainset,
        method_cfg=seeded_config["method"],
    )
    if hasattr(baseline, "clear_diagnostics"):
        baseline.clear_diagnostics()

    variant_counterfactuals = {
        variant_cfg.name: []
        for variant_cfg in _variant_configs(budget, iterations, variant_scope)
    }
    ground_truth_rows: list[pd.DataFrame] = []
    row_traces: list[dict[str, object]] = []
    variant_cfgs = _variant_configs(budget, iterations, variant_scope)
    factual_iterator = tqdm(
        list(factuals.iterrows()),
        desc=f"seed {seed} factuals",
        leave=False,
    )
    for row_index, (_, row) in enumerate(factual_iterator):
        factual = row.to_frame().T.loc[:, schema.feature_names].reset_index(drop=True)
        row_seed = int(seed) + row_index
        factual_iterator.set_postfix_str(
            f"row_seed={row_seed} variants={len(variant_cfgs)}"
        )

        with seed_context(row_seed):
            metric_user = _build_user(
                factual=factual,
                schema=schema,
                gt_generator=gt_generator,
                method_cfg=seeded_config["method"],
                seed=row_seed,
            )
            base_counterfactual = baseline.generate(factual).reset_index(drop=True)
        ground_truth_rows.append(metric_user.ground_truth)
        variant_counterfactuals["base"].append(base_counterfactual)
        should_trace_row = (
            trace_seed is not None
            and int(trace_seed) == int(seed)
            and len(row_traces) < int(trace_limit)
        )
        row_trace: dict[str, object] | None = None
        if should_trace_row:
            row_trace = {
                "row_index": int(row_index),
                "row_seed": int(row_seed),
                "factual": _trace_dataframe(factual),
                "ground_truth": _trace_dataframe(metric_user.ground_truth),
                "base_counterfactual": _trace_dataframe(base_counterfactual),
                "base_metrics": _trace_counterfactual_metrics(
                    factual,
                    metric_user.ground_truth,
                    base_counterfactual,
                ),
                "variants": {},
            }

        variant_iterator = tqdm(
            [cfg for cfg in variant_cfgs if cfg.name != "base"],
            desc=f"seed {seed} row {row_index + 1}/{factuals.shape[0]} variants",
            leave=False,
        )
        for variant_cfg in variant_iterator:
            variant_iterator.set_postfix_str(VARIANT_LABELS[variant_cfg.name])
            trace_steps = [] if row_trace is not None else None
            with seed_context(row_seed):
                counterfactual = _run_search_variant(
                    factual=factual,
                    user=metric_user,
                    initial_counterfactual=base_counterfactual,
                    adapter=adapter,
                    target_model=local_model,
                    schema=schema,
                    method_cfg=seeded_config["method"],
                    round_query_budgets=(
                        _resolve_reference_multi_round_query_budgets(
                            budget=variant_cfg.budget,
                            candidate_count=variant_cfg.candidate_count,
                        )
                        if variant_cfg.candidate_count is not None
                        else _resolve_round_query_budgets(
                            budget=variant_cfg.budget,
                            iterations=variant_cfg.iterations,
                        )
                    ),
                    selection=variant_cfg.selection,
                    trace_steps=trace_steps,
                )
            variant_counterfactuals[variant_cfg.name].append(counterfactual)
            if row_trace is not None:
                row_trace["variants"][variant_cfg.name] = {
                    "counterfactual": _trace_dataframe(counterfactual),
                    "metrics": _trace_counterfactual_metrics(
                        factual,
                        metric_user.ground_truth,
                        counterfactual,
                    ),
                    "steps": trace_steps,
                }

        if row_trace is not None:
            row_traces.append(row_trace)

    ground_truths = pd.concat(ground_truth_rows, axis=0, ignore_index=True)
    desired_index = int(local_model.get_class_to_index()[desired_class])
    seed_metrics: dict[str, dict[str, float]] = {}
    for variant_name, rows in variant_counterfactuals.items():
        counterfactuals = pd.concat(rows, axis=0, ignore_index=True)
        seed_metrics[variant_name] = _compute_variant_metrics(
            factuals=factuals.reset_index(drop=True),
            ground_truths=ground_truths,
            counterfactuals=counterfactuals,
            schema=schema,
            adapter=adapter,
            desired_index=desired_index,
        )
        seed_metrics[variant_name]["valid_count"] = float(_count_valid_rows(counterfactuals))
    base_search_diagnostics = (
        baseline.diagnostics() if hasattr(baseline, "diagnostics") else {}
    )
    if base_search_diagnostics:
        print(f"[seed {seed}] base_search={base_search_diagnostics}")
    metric_debug_rows: list[dict[str, object]] = []
    if metric_debug and "base" in variant_counterfactuals:
        metric_debug_rows = _collect_metric_debug_rows(
            factuals=factuals.reset_index(drop=True),
            ground_truths=ground_truths,
            counterfactuals=pd.concat(
                variant_counterfactuals["base"], axis=0, ignore_index=True
            ),
            limit=metric_debug_limit,
        )
        if metric_debug_rows:
            print(
                f"[seed {seed}] metric_debug_base="
                f"{json.dumps(metric_debug_rows, indent=2)}"
            )

    return {
        "seed": int(seed),
        "num_factuals": int(factuals.shape[0]),
        "negatives_available": int(diagnostics["negatives_available"]),
        "model_diagnostics": diagnostics,
        "base_search_diagnostics": base_search_diagnostics,
        "metric_debug_base": metric_debug_rows,
        "trace_rows": row_traces,
        "metrics": seed_metrics,
    }


def _run_model_reproduction(
    config_path: Path,
    device: str,
    seeds: list[int],
    num_instances: int,
    budget: int,
    iterations: int,
    ground_truth_scale: str,
    compare_paper_targets: bool,
    variant_scope: str,
    train_balance: str,
    metric_debug: bool,
    metric_debug_limit: int,
    base_lambda_override: float | None,
    base_n_iter_override: int | None,
    base_t_max_min_override: float | None,
    trace_seed: int | None,
    trace_limit: int,
    trace_output: str | None,
) -> dict[str, object]:
    base_config = _apply_carla_reference_defaults(
        _apply_device(_load_config(config_path), device)
    )
    if base_lambda_override is not None:
        base_config["method"]["baseline_lambda"] = float(base_lambda_override)
    if base_n_iter_override is not None:
        base_config["method"]["baseline_n_iter"] = int(base_n_iter_override)
    if base_t_max_min_override is not None:
        base_config["method"]["baseline_t_max_min"] = float(base_t_max_min_override)
    per_seed_results = []
    seed_iterator = tqdm(
        seeds,
        desc="ANN seeds",
    )
    for seed in seed_iterator:
        seed_iterator.set_postfix_str(
            f"scope={variant_scope} instances={num_instances} budget={budget} iterations={iterations}"
        )
        per_seed_results.append(
            _run_seed(
                config=base_config,
                seed=seed,
                num_instances=num_instances,
                budget=budget,
                iterations=iterations,
                ground_truth_scale=ground_truth_scale,
                variant_scope=variant_scope,
                train_balance=train_balance,
                metric_debug=metric_debug,
                metric_debug_limit=metric_debug_limit,
                trace_seed=trace_seed,
                trace_limit=trace_limit,
            )
        )

    seed_metrics: dict[str, list[dict[str, float]]] = {
        variant_cfg.name: []
        for variant_cfg in _variant_configs(budget, iterations, variant_scope)
    }
    for result in per_seed_results:
        metrics = result["metrics"]
        for variant_name in seed_metrics:
            seed_metrics[variant_name].append(metrics[variant_name])

    aggregates = _aggregate_seed_metrics(seed_metrics)
    table1 = _table1_dataframe(aggregates, compare_paper_targets)
    output: dict[str, object] = {
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "seeds": [int(seed) for seed in seeds],
        "device": device,
        "num_factuals_per_seed": [int(result["num_factuals"]) for result in per_seed_results],
        "per_seed": per_seed_results,
        "aggregates": aggregates,
        "table1": table1.to_dict(orient="records"),
        "trace_seed": None if trace_seed is None else int(trace_seed),
        "trace_limit": int(trace_limit),
        "traces": [
            {
                "seed": int(result["seed"]),
                "trace_rows": result.get("trace_rows", []),
            }
            for result in per_seed_results
            if result.get("trace_rows")
        ],
    }
    table4 = _table4_dataframe(aggregates, compare_paper_targets)
    output["table4"] = table4.to_dict(orient="records")

    print("\n=== ANN COMPAS + Wachter (Table 1 target) ===")
    print(table1.to_string(index=False))
    print("\n=== ANN COMPAS + Wachter (Table 4 target) ===")
    print(_table4_dataframe(aggregates, compare_paper_targets).to_string(index=False))
    if trace_output:
        trace_path = Path(trace_output)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(output["traces"], indent=2), encoding="utf-8")
        print(f"\n=== Trace Output ===\n{trace_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_ANN_CONFIG))
    parser.add_argument(
        "--variant-scope",
        choices=["base", "trace_base", "all"],
        default="all",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
    )
    parser.add_argument("--num-instances", type=int, default=100)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--ground-truth-scale",
        choices=["near", "intermediate", "far"],
        default="far",
    )
    parser.add_argument(
        "--compare-paper-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--train-balance",
        choices=["auto", "rebalance", "off"],
        default="auto",
    )
    parser.add_argument(
        "--metric-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--metric-debug-limit",
        type=int,
        default=2,
    )
    parser.add_argument("--base-lambda", type=float, default=None)
    parser.add_argument("--base-n-iter", type=int, default=None)
    parser.add_argument("--base-t-max-min", type=float, default=None)
    parser.add_argument("--trace-seed", type=int, default=None)
    parser.add_argument("--trace-limit", type=int, default=3)
    parser.add_argument("--trace-output", type=str, default=None)
    args = parser.parse_args()

    if args.num_instances < 1:
        raise ValueError("num-instances must be >= 1")
    if args.budget < 1:
        raise ValueError("budget must be >= 1")
    if args.iterations < 1:
        raise ValueError("iterations must be >= 1")
    if args.metric_debug_limit < 0:
        raise ValueError("metric-debug-limit must be >= 0")
    if args.base_n_iter is not None and args.base_n_iter < 1:
        raise ValueError("base-n-iter must be >= 1")
    if args.base_lambda is not None and args.base_lambda < 0:
        raise ValueError("base-lambda must be >= 0")
    if args.base_t_max_min is not None and args.base_t_max_min <= 0:
        raise ValueError("base-t-max-min must be > 0")
    if args.trace_limit < 0:
        raise ValueError("trace-limit must be >= 0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = (PROJECT_ROOT / args.config).resolve()
    output = _run_model_reproduction(
        config_path=config_path,
        device=device,
        seeds=list(args.seeds),
        num_instances=int(args.num_instances),
        budget=int(args.budget),
        iterations=int(args.iterations),
        ground_truth_scale=args.ground_truth_scale,
        compare_paper_targets=bool(args.compare_paper_targets),
        variant_scope=args.variant_scope,
        train_balance=args.train_balance,
        metric_debug=bool(args.metric_debug),
        metric_debug_limit=int(args.metric_debug_limit),
        base_lambda_override=args.base_lambda,
        base_n_iter_override=args.base_n_iter,
        base_t_max_min_override=args.base_t_max_min,
        trace_seed=args.trace_seed,
        trace_limit=int(args.trace_limit),
        trace_output=args.trace_output,
    )

    print("\n=== JSON Summary ===")
    print(json.dumps({"ann": output}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
