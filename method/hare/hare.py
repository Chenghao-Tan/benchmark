from __future__ import annotations

import random

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.hare.baselines import build_baseline_generator
from method.hare.oracles import SimulatedUser, select_best_candidate
from method.hare.sampling import actionable_sampling
from method.hare.search import boundary_point_search, calibrate_candidate_set
from method.hare.support import (
    FeatureSchema,
    ModelAdapter,
    resolve_feature_schema,
    resolve_target_indices,
    validate_counterfactuals,
    validate_numeric_frame,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("hare")
class HareMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        base_method: str = "wachter",
        budget: int = 30,
        iterations: int = 1,
        candidate_count: int | None = None,
        sampling_epochs: int = 100,
        sampling_lr: float = 0.1,
        sampling_lambda: float = 10.0,
        sampling_radius: float = 1.0,
        boundary_epsilon: float = 1e-6,
        calibrate_all_candidates: bool = True,
        ground_truth_scale: str = "far",
        use_preferences: bool = False,
        noise_prob: float = 0.0,
        selection_strategy: str = "oracle",
        bs_final_only: bool = False,
        baseline_lr: float = 0.01,
        baseline_lambda: float = 0.1,
        baseline_n_iter: int = 2500,
        baseline_t_max_min: float = 0.7,
        baseline_norm: int | None = None,
        baseline_loss_type: str = "BCE",
        baseline_n_search_samples: int = 1000,
        baseline_step: float = 0.2,
        baseline_max_iter: int = 1000,
        baseline_fraction: float = 0.1,
        baseline_mode: str = "knn",
        baseline_n_neighbors: int = 50,
        baseline_radius: float = 0.25,
        gt_n_search_samples: int = 300,
        gt_step: float = 0.2,
        gt_max_iter: int = 1000,
        gt_max_attempts: int = 64,
        **kwargs,
    ):
        del kwargs
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = str(base_method).lower() == "wachter"
        self._is_trained = False
        self._desired_class = desired_class

        self._base_method = str(base_method).lower()
        self._budget = int(budget)
        self._iterations = int(iterations)
        self._candidate_count = None if candidate_count is None else int(candidate_count)
        self._sampling_epochs = int(sampling_epochs)
        self._sampling_lr = float(sampling_lr)
        self._sampling_lambda = float(sampling_lambda)
        self._sampling_radius = float(sampling_radius)
        self._boundary_epsilon = float(boundary_epsilon)
        self._calibrate_all_candidates = bool(calibrate_all_candidates)
        self._ground_truth_scale = str(ground_truth_scale).lower()
        self._use_preferences = bool(use_preferences)
        self._noise_prob = float(noise_prob)
        self._selection_strategy = str(selection_strategy).lower()
        self._bs_final_only = bool(bs_final_only)

        self._baseline_lr = float(baseline_lr)
        self._baseline_lambda = float(baseline_lambda)
        self._baseline_n_iter = int(baseline_n_iter)
        self._baseline_t_max_min = float(baseline_t_max_min)
        self._baseline_norm = None if baseline_norm is None else int(baseline_norm)
        self._baseline_loss_type = str(baseline_loss_type).upper()
        self._baseline_n_search_samples = int(baseline_n_search_samples)
        self._baseline_step = float(baseline_step)
        self._baseline_max_iter = int(baseline_max_iter)
        self._baseline_fraction = float(baseline_fraction)
        self._baseline_mode = str(baseline_mode).lower()
        self._baseline_n_neighbors = int(baseline_n_neighbors)
        self._baseline_radius = float(baseline_radius)

        self._gt_n_search_samples = int(gt_n_search_samples)
        self._gt_step = float(gt_step)
        self._gt_max_iter = int(gt_max_iter)
        self._gt_max_attempts = int(gt_max_attempts)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._budget < 1:
            raise ValueError("budget must be >= 1")
        if self._iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self._candidate_count is not None and self._candidate_count < 1:
            raise ValueError("candidate_count must be >= 1 when provided")
        if self._sampling_epochs < 1:
            raise ValueError("sampling_epochs must be >= 1")
        if self._sampling_lr <= 0:
            raise ValueError("sampling_lr must be > 0")
        if self._sampling_radius <= 0:
            raise ValueError("sampling_radius must be > 0")
        if self._boundary_epsilon <= 0:
            raise ValueError("boundary_epsilon must be > 0")
        if self._baseline_n_iter < 1:
            raise ValueError("baseline_n_iter must be >= 1")
        if self._baseline_n_search_samples < 1:
            raise ValueError("baseline_n_search_samples must be >= 1")
        if self._baseline_step <= 0:
            raise ValueError("baseline_step must be > 0")
        if self._baseline_max_iter < 1:
            raise ValueError("baseline_max_iter must be >= 1")
        if self._gt_n_search_samples < 1:
            raise ValueError("gt_n_search_samples must be >= 1")
        if self._gt_step <= 0:
            raise ValueError("gt_step must be > 0")
        if self._gt_max_iter < 1:
            raise ValueError("gt_max_iter must be >= 1")
        if self._gt_max_attempts < 1:
            raise ValueError("gt_max_attempts must be >= 1")
        if self._base_method not in {"wachter", "gs", "growing_spheres", "face"}:
            raise ValueError("base_method must be one of {'wachter', 'gs', 'face'}")
        if self._ground_truth_scale not in {"near", "intermediate", "far"}:
            raise ValueError(
                "ground_truth_scale must be one of {'near', 'intermediate', 'far'}"
            )
        if self._selection_strategy not in {"oracle", "random"}:
            raise ValueError("selection_strategy must be one of {'oracle', 'random'}")

    def _resolve_round_query_budgets(self) -> list[int]:
        if self._candidate_count is not None:
            return [self._candidate_count for _ in range(self._iterations)]

        active_rounds = min(self._iterations, self._budget)
        base_budget = self._budget // active_rounds
        remainder = self._budget % active_rounds
        return [
            base_budget + (1 if round_index < remainder else 0)
            for round_index in range(active_rounds)
        ]

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for HareMethod.fit()")

        with seed_context(self._seed):
            features = trainset.get(target=False)
            validate_numeric_frame(features, "HareMethod")
            class_to_index = self._target_model.get_class_to_index()
            if len(class_to_index) != 2:
                raise ValueError("HARE currently supports binary classification only")

            self._schema: FeatureSchema = resolve_feature_schema(trainset)
            self._feature_names = list(self._schema.feature_names)
            self._adapter = ModelAdapter(self._target_model, self._feature_names)
            self._round_query_budgets = self._resolve_round_query_budgets()

            self._baseline = build_baseline_generator(
                base_method=self._base_method,
                target_model=self._target_model,
                schema=self._schema,
                desired_class=self._desired_class,
                trainset=trainset,
                baseline_lr=self._baseline_lr,
                baseline_lambda=self._baseline_lambda,
                baseline_n_iter=self._baseline_n_iter,
                baseline_t_max_min=self._baseline_t_max_min,
                baseline_norm=self._baseline_norm,
                baseline_loss_type=self._baseline_loss_type,
                baseline_n_search_samples=self._baseline_n_search_samples,
                baseline_step=self._baseline_step,
                baseline_max_iter=self._baseline_max_iter,
                baseline_fraction=self._baseline_fraction,
                baseline_mode=self._baseline_mode,
                baseline_n_neighbors=self._baseline_n_neighbors,
                baseline_radius=self._baseline_radius,
                baseline_seed=self._seed,
            )
            self._gt_generator = build_baseline_generator(
                base_method="gs",
                target_model=self._target_model,
                schema=self._schema,
                desired_class=self._desired_class,
                trainset=trainset,
                baseline_n_search_samples=self._gt_n_search_samples,
                baseline_step=self._gt_step,
                baseline_max_iter=self._gt_max_iter,
                baseline_norm=1,
                baseline_seed=self._seed,
            )
            self._is_trained = True

    def _target_index(self, factual: pd.DataFrame) -> int:
        original_prediction = self._adapter.predict_label_indices(factual)
        return int(
            resolve_target_indices(
                target_model=self._target_model,
                original_prediction=original_prediction,
                desired_class=self._desired_class,
            )[0]
        )

    def _generate_row_counterfactual(self, factual: pd.DataFrame) -> np.ndarray:
        current_counterfactual = self._baseline.generate(factual)
        if current_counterfactual.isna().any(axis=1).iloc[0]:
            row = np.empty(len(self._feature_names), dtype=np.float32)
            row[:] = np.nan
            return row

        factual = factual.loc[:, self._feature_names].copy(deep=True).reset_index(drop=True)
        current_counterfactual = current_counterfactual.reset_index(drop=True)
        target_index = self._target_index(factual)
        user = SimulatedUser(
            factual=factual,
            schema=self._schema,
            gt_generator=self._gt_generator,
            ground_truth_scale=self._ground_truth_scale,
            use_preferences=self._use_preferences,
            noise_prob=self._noise_prob,
            max_attempts=self._gt_max_attempts,
            seed=self._seed,
        )

        for round_budget in self._round_query_budgets:
            candidates = actionable_sampling(
                baseline=current_counterfactual,
                factual=factual,
                model=self._adapter,
                schema=self._schema,
                target_index=target_index,
                num_candidates=round_budget,
                radius=self._sampling_radius,
                lambda_=self._sampling_lambda,
                lr=self._sampling_lr,
                epochs=self._sampling_epochs,
            )
            apply_binary_search_to_all = (
                self._calibrate_all_candidates and not self._bs_final_only
            )
            if apply_binary_search_to_all:
                candidates = calibrate_candidate_set(
                    factual=factual,
                    candidates=candidates,
                    model=self._adapter,
                    target_index=target_index,
                    epsilon=self._boundary_epsilon,
                )
                candidates = validate_counterfactuals(
                    target_model=self._target_model,
                    factuals=pd.concat([factual] * candidates.shape[0], ignore_index=True),
                    candidates=candidates,
                    desired_class=self._desired_class,
                )
                candidates = candidates.dropna(axis=0, how="any").reset_index(drop=True)

            if candidates.shape[0] == 0:
                break

            if self._selection_strategy == "random":
                winner_index = random.randint(0, candidates.shape[0] - 1)
            else:
                winner_index = select_best_candidate(user, candidates)
            current_counterfactual = candidates.iloc[[winner_index]].copy(deep=True)

            if self._bs_final_only or not self._calibrate_all_candidates:
                current_counterfactual = boundary_point_search(
                    factual=factual,
                    candidate=current_counterfactual,
                    model=self._adapter,
                    target_index=target_index,
                    epsilon=self._boundary_epsilon,
                )

        return current_counterfactual.iloc[0].to_numpy(dtype="float32")

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        rows: list[np.ndarray] = []
        for row_index, (_, row) in enumerate(factuals.iterrows()):
            row_seed = None if self._seed is None else self._seed + row_index
            with seed_context(row_seed):
                row_df = row.to_frame().T.loc[:, self._feature_names]
                rows.append(self._generate_row_counterfactual(row_df))

        candidates = pd.DataFrame(rows, index=factuals.index, columns=self._feature_names)
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )
