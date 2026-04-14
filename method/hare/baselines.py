from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csgraph, csr_matrix
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

from dataset.dataset_object import DatasetObject
from method.hare.support import (
    BlackBoxModelTypes,
    FeatureSchema,
    ModelAdapter,
    TorchModelTypes,
    actionable_mask,
    ensure_supported_target_model,
    resolve_target_indices,
    validate_counterfactuals,
)
from model.model_object import ModelObject
from utils.seed import seed_context


class InternalBaseline:
    def __init__(
        self,
        target_model: ModelObject,
        schema: FeatureSchema,
        desired_class: int | str | None = None,
    ) -> None:
        self._target_model = target_model
        self._schema = schema
        self._desired_class = desired_class
        self._adapter = ModelAdapter(target_model, schema.feature_names)

    def fit(self, trainset: DatasetObject | None) -> None:
        del trainset

    def _nan_row(self) -> np.ndarray:
        row = np.empty(len(self._schema.feature_names), dtype=np.float32)
        row[:] = np.nan
        return row

    def _target_index(self, factual: pd.DataFrame) -> int:
        original_prediction = self._adapter.predict_label_indices(factual)
        return int(
            resolve_target_indices(
                target_model=self._target_model,
                original_prediction=original_prediction,
                desired_class=self._desired_class,
            )[0]
        )

    def _predict_label_index(self, values: np.ndarray) -> int:
        return int(self._adapter.predict_label_indices(values.reshape(1, -1))[0])

    def _generate_row(self, factual: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def generate(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = factuals.loc[:, self._schema.feature_names].copy(deep=True)
        rows = [self._generate_row(row.to_frame().T) for _, row in factuals.iterrows()]
        candidates = pd.DataFrame(rows, index=factuals.index, columns=self._schema.feature_names)
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )


class WachterBaseline(InternalBaseline):
    def __init__(
        self,
        target_model: ModelObject,
        schema: FeatureSchema,
        desired_class: int | str | None = None,
        lr: float = 0.01,
        lambda_: float = 0.1,
        n_iter: int = 2500,
        t_max_min: float = 0.7,
        norm: int = 1,
        loss_type: str = "BCE",
    ) -> None:
        ensure_supported_target_model(target_model, TorchModelTypes, "WachterBaseline")
        super().__init__(target_model=target_model, schema=schema, desired_class=desired_class)
        self._lr = float(lr)
        self._lambda = float(lambda_)
        self._n_iter = int(n_iter)
        self._t_max_seconds = float(t_max_min) * 60.0
        self._norm = int(norm)
        self._loss_type = str(loss_type).upper()
        self._row_diagnostics: list[dict[str, float | bool]] = []

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
            "base_lambda_reduced_fraction": float(
                (lambda_reductions > 0.0).mean()
            ),
            "base_target_probability_mean": float(target_probabilities.mean()),
            "base_target_probability_min": float(target_probabilities.min()),
            "base_target_probability_max": float(target_probabilities.max()),
            "base_internal_success_fraction": float(successes.mean()),
        }

    def _reconstruct_encoding_constraints(self, values: torch.Tensor) -> torch.Tensor:
        reconstructed = values.clone()
        for index in self._schema.binary_indices:
            reconstructed[:, index] = torch.round(reconstructed[:, index])
        return reconstructed

    def _generate_row(self, factual: pd.DataFrame) -> np.ndarray:
        torch.manual_seed(0)
        factual_values = factual.to_numpy(dtype="float32").reshape(-1)
        factual_tensor = torch.tensor(
            factual_values.reshape(1, -1),
            dtype=torch.float32,
            device=self._target_model._device,
        )
        candidate = factual_tensor.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([candidate], lr=self._lr, amsgrad=True)
        target_index = self._target_index(factual)
        target_vector = torch.zeros(
            len(self._target_model.get_class_to_index()),
            dtype=torch.float32,
            device=self._target_model._device,
        )
        target_vector[target_index] = 1.0
        lambda_param = torch.tensor(
            self._lambda,
            dtype=torch.float32,
            device=self._target_model._device,
        )
        current_probability = torch.tensor(
            [0.0],
            dtype=torch.float32,
            device=self._target_model._device,
        )
        start_time = time.monotonic()
        total_iterations = 0
        lambda_reductions = 0

        while float(current_probability.item()) <= 0.5:
            iteration = 0
            while (
                float(current_probability.item()) <= 0.5
                and iteration < max(1, self._n_iter)
            ):
                optimizer.zero_grad()
                projected_candidate = self._reconstruct_encoding_constraints(candidate)
                probabilities = self._adapter.predict_proba(projected_candidate)
                if not isinstance(probabilities, torch.Tensor):
                    return self._nan_row()

                current_probability = probabilities[:, target_index]
                if self._loss_type == "MSE":
                    clipped_probability = current_probability.clamp(
                        min=1e-6,
                        max=1.0 - 1e-6,
                    )
                    classification_input = torch.log(
                        clipped_probability / (1.0 - clipped_probability)
                    )
                    target_loss = torch.tensor(
                        [1.0],
                        dtype=torch.float32,
                        device=self._target_model._device,
                    )
                else:
                    classification_input = probabilities.squeeze(0)
                    target_loss = target_vector

                cost = torch.linalg.norm(
                    (projected_candidate - factual_tensor).reshape(-1),
                    ord=self._norm,
                )
                loss = F.binary_cross_entropy(
                    classification_input.clamp(min=1e-6, max=1.0 - 1e-6),
                    target_loss,
                ) + lambda_param * cost
                loss.backward()
                if candidate.grad is not None and self._schema.immutable_indices:
                    candidate.grad[:, self._schema.immutable_indices] = 0.0
                optimizer.step()

                with torch.no_grad():
                    candidate.clamp_(0.0, 1.0)
                iteration += 1
                total_iterations += 1

            if float(current_probability.item()) <= 0.5:
                lambda_reductions += 1
            lambda_param = torch.clamp(lambda_param / 2.0, min=0.0)
            if time.monotonic() - start_time > self._t_max_seconds:
                break
            if float(current_probability.item()) >= 0.5:
                break

        projected_candidate = self._reconstruct_encoding_constraints(candidate)
        output = projected_candidate.detach().cpu().numpy().reshape(-1)
        final_probability = float(
            self._adapter.predict_proba(projected_candidate)[:, target_index]
            .detach()
            .cpu()
            .numpy()[0]
        )
        self._row_diagnostics.append(
            {
                "iterations": float(total_iterations),
                "lambda_reductions": float(lambda_reductions),
                "target_probability": final_probability,
                "success": bool(final_probability > 0.5),
            }
        )
        return output


class GrowingSpheresBaseline(InternalBaseline):
    def __init__(
        self,
        target_model: ModelObject,
        schema: FeatureSchema,
        desired_class: int | str | None = None,
        n_search_samples: int = 1000,
        p_norm: int = 1,
        step: float = 0.2,
        max_iter: int = 1000,
        seed: int | None = None,
    ) -> None:
        ensure_supported_target_model(
            target_model,
            BlackBoxModelTypes,
            "GrowingSpheresBaseline",
        )
        super().__init__(target_model=target_model, schema=schema, desired_class=desired_class)
        self._n_search_samples = int(n_search_samples)
        self._p_norm = int(p_norm)
        self._step = float(step)
        self._max_iter = int(max_iter)
        self._seed = seed

    def _generate_row(self, factual: pd.DataFrame) -> np.ndarray:
        factual_values = factual.to_numpy(dtype="float32").reshape(-1)
        target_index = self._target_index(factual)

        if not self._schema.mutable_indices:
            return self._nan_row()

        low = 0.0
        high = low + self._step
        best_candidate = self._nan_row()
        mutable_binary_indices = [
            index
            for index in self._schema.mutable_indices
            if index not in self._schema.mutable_continuous_indices
        ]

        for _ in range(max(1, self._max_iter)):
            candidates = np.repeat(
                factual_values.reshape(1, -1),
                self._n_search_samples,
                axis=0,
            ).astype(np.float32)

            with seed_context(self._seed):
                if self._schema.mutable_continuous_indices:
                    noise = np.random.randn(
                        self._n_search_samples,
                        len(self._schema.mutable_continuous_indices),
                    ).astype(np.float32)
                    norms = np.linalg.norm(noise, ord=self._p_norm, axis=1)
                    dist = (
                        np.random.rand(self._n_search_samples).astype(np.float32)
                        * (high - low)
                        + low
                    )
                    scale = np.divide(
                        dist,
                        norms,
                        out=np.zeros_like(dist),
                        where=norms != 0.0,
                    ).reshape(-1, 1)
                    candidates[:, self._schema.mutable_continuous_indices] = (
                        factual_values[self._schema.mutable_continuous_indices].reshape(1, -1)
                        + noise * scale
                    )

                if mutable_binary_indices:
                    _ = np.random.binomial(
                        1,
                        0.5,
                        size=self._n_search_samples * len(mutable_binary_indices),
                    ).reshape(self._n_search_samples, -1)

            predictions = self._adapter.predict_label_indices(
                pd.DataFrame(candidates, columns=self._schema.feature_names)
            )
            successful = candidates[predictions == target_index]
            if successful.shape[0] > 0:
                deltas = successful - factual_values.reshape(1, -1)
                if self._p_norm == 1:
                    distances = np.abs(deltas).sum(axis=1)
                elif self._p_norm == 2:
                    distances = np.square(deltas).sum(axis=1)
                else:
                    raise ValueError("Distance not defined yet")
                best_candidate = successful[int(np.argmin(distances))]
                break

            low = high
            high = low + self._step

        return best_candidate


@dataclass
class FaceGraphConfig:
    mode: str = "knn"
    fraction: float = 0.1
    n_neighbors: int = 50
    radius: float = 0.25
    p_norm: int = 2


class FaceBaseline(InternalBaseline):
    def __init__(
        self,
        target_model: ModelObject,
        schema: FeatureSchema,
        desired_class: int | str | None = None,
        mode: str = "knn",
        fraction: float = 0.1,
        n_neighbors: int = 50,
        radius: float = 0.25,
        p_norm: int = 2,
    ) -> None:
        ensure_supported_target_model(target_model, BlackBoxModelTypes, "FaceBaseline")
        super().__init__(target_model=target_model, schema=schema, desired_class=desired_class)
        self._cfg = FaceGraphConfig(
            mode=str(mode).lower(),
            fraction=float(fraction),
            n_neighbors=int(n_neighbors),
            radius=float(radius),
            p_norm=int(p_norm),
        )
        self._train_features: pd.DataFrame | None = None

    def fit(self, trainset: DatasetObject | None) -> None:
        if trainset is None:
            raise ValueError("trainset is required for FaceBaseline.fit()")
        self._train_features = trainset.get(target=False).loc[:, self._schema.feature_names].copy(
            deep=True
        )

    def _choose_subset(self) -> pd.DataFrame:
        if self._train_features is None:
            raise RuntimeError("FaceBaseline has not been fitted")

        if self._cfg.fraction >= 1.0:
            return self._train_features.copy(deep=True)

        sample_size = int(np.rint(self._cfg.fraction * self._train_features.shape[0]))
        sample_size = max(1, min(sample_size, self._train_features.shape[0]))
        chosen = np.random.choice(
            self._train_features.index.to_numpy(),
            size=sample_size,
            replace=False,
        )
        return self._train_features.loc[chosen].copy(deep=True)

    def _build_constraint_matrix(self, data_values: np.ndarray) -> np.ndarray:
        if not self._schema.immutable_indices:
            return np.ones((data_values.shape[0], data_values.shape[0]), dtype=np.float32)

        constraint = np.ones((data_values.shape[0], data_values.shape[0]), dtype=np.float32)
        for index in self._schema.immutable_indices:
            column = data_values[:, index]
            equality = np.isclose(
                column.reshape(-1, 1),
                column.reshape(1, -1),
                atol=1e-6,
                rtol=0.0,
            ).astype(np.float32)
            constraint *= equality
        return constraint

    def _build_graph(self, data_values: np.ndarray, constraint: np.ndarray) -> csr_matrix:
        if self._cfg.mode == "knn":
            neighbors = min(max(1, self._cfg.n_neighbors), max(1, data_values.shape[0] - 1))
            graph = kneighbors_graph(data_values, n_neighbors=neighbors, mode="distance")
        elif self._cfg.mode == "epsilon":
            graph = radius_neighbors_graph(
                data_values,
                radius=self._cfg.radius,
                mode="distance",
            )
        else:
            raise ValueError("FaceBaseline mode must be one of {'knn', 'epsilon'}")

        adjacency = graph.toarray().astype(np.float32)
        adjacency *= constraint
        return csr_matrix(adjacency)

    def _closest_actionable_training_point(
        self,
        factual_values: np.ndarray,
        subset: pd.DataFrame,
        target_index: int,
    ) -> np.ndarray:
        subset_values = subset.to_numpy(dtype="float32")
        predictions = self._adapter.predict_label_indices(subset)
        mask = actionable_mask(subset_values, factual_values, self._schema)
        candidate_values = subset_values[(predictions == target_index) & mask]
        if candidate_values.shape[0] == 0:
            return self._nan_row()

        distances = np.linalg.norm(
            candidate_values - factual_values.reshape(1, -1),
            ord=self._cfg.p_norm,
            axis=1,
        )
        return candidate_values[int(np.argmin(distances))]

    def _generate_row(self, factual: pd.DataFrame) -> np.ndarray:
        factual_values = factual.to_numpy(dtype="float32").reshape(-1)
        target_index = self._target_index(factual)
        subset = self._choose_subset()
        data = pd.concat([factual, subset], axis=0, ignore_index=True)
        data_values = data.to_numpy(dtype="float32")

        predictions = self._adapter.predict_label_indices(data)
        actionability = actionable_mask(data_values, factual_values, self._schema)
        positive_indices = np.where(
            (predictions == target_index) & actionability & (np.arange(data.shape[0]) != 0)
        )[0]
        if positive_indices.size == 0:
            return self._closest_actionable_training_point(factual_values, subset, target_index)

        graph = self._build_graph(
            data_values=data_values,
            constraint=self._build_constraint_matrix(data_values),
        )
        distances = csgraph.dijkstra(
            csgraph=graph,
            directed=False,
            indices=0,
            return_predecessors=False,
        )
        reachable_indices = positive_indices[np.isfinite(distances[positive_indices])]
        if reachable_indices.size == 0:
            return self._closest_actionable_training_point(factual_values, subset, target_index)

        graph_distances = distances[reachable_indices]
        best_graph_distance = graph_distances.min()
        shortlisted = reachable_indices[
            np.isclose(graph_distances, best_graph_distance, atol=1e-6, rtol=0.0)
        ]
        shortlisted_values = data_values[shortlisted]
        lpnorm = np.linalg.norm(
            shortlisted_values - factual_values.reshape(1, -1),
            ord=self._cfg.p_norm,
            axis=1,
        )
        return shortlisted_values[int(np.argmin(lpnorm))]


def build_baseline_generator(
    base_method: str,
    target_model: ModelObject,
    schema: FeatureSchema,
    desired_class: int | str | None = None,
    trainset: DatasetObject | None = None,
    baseline_lr: float = 0.01,
    baseline_lambda: float = 0.1,
    baseline_n_iter: int = 2500,
    baseline_t_max_min: float = 0.7,
    baseline_norm: int | None = 1,
    baseline_loss_type: str = "BCE",
    baseline_n_search_samples: int = 1000,
    baseline_step: float = 0.2,
    baseline_max_iter: int = 1000,
    baseline_fraction: float = 0.1,
    baseline_mode: str = "knn",
    baseline_n_neighbors: int = 50,
    baseline_radius: float = 0.25,
    baseline_seed: int | None = None,
) -> InternalBaseline:
    method_name = str(base_method).lower()
    resolved_norm = baseline_norm
    if method_name == "wachter":
        if resolved_norm is None:
            resolved_norm = 1
        generator: InternalBaseline = WachterBaseline(
            target_model=target_model,
            schema=schema,
            desired_class=desired_class,
            lr=baseline_lr,
            lambda_=baseline_lambda,
            n_iter=baseline_n_iter,
            t_max_min=baseline_t_max_min,
            norm=resolved_norm,
            loss_type=baseline_loss_type,
        )
    elif method_name in {"gs", "growing_spheres"}:
        if resolved_norm is None:
            resolved_norm = 2
        generator = GrowingSpheresBaseline(
            target_model=target_model,
            schema=schema,
            desired_class=desired_class,
            n_search_samples=baseline_n_search_samples,
            p_norm=resolved_norm,
            step=baseline_step,
            max_iter=baseline_max_iter,
            seed=baseline_seed,
        )
    elif method_name == "face":
        if resolved_norm is None:
            resolved_norm = 2
        generator = FaceBaseline(
            target_model=target_model,
            schema=schema,
            desired_class=desired_class,
            mode=baseline_mode,
            fraction=baseline_fraction,
            n_neighbors=baseline_n_neighbors,
            radius=baseline_radius,
            p_norm=resolved_norm,
        )
    else:
        raise ValueError("base_method must be one of {'wachter', 'gs', 'face'}")

    generator.fit(trainset)
    return generator
