from __future__ import annotations

import numpy as np
import pandas as pd

from method.hare.support import FeatureSchema, project_actionability_array
from utils.seed import seed_context

_GROUND_TRUTH_RADIUS = {
    "near": 0.2,
    "intermediate": 0.8,
    "far": 1.5,
}


def sample_preference_weights(dimension: int) -> np.ndarray:
    centers = np.array([0.2, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    means = np.random.choice(centers, size=int(dimension))
    preference = np.random.multivariate_normal(means, cov=np.eye(int(dimension)))
    return np.clip(preference.astype(np.float32), a_min=0.1, a_max=5.0)


class SimulatedUser:
    def __init__(
        self,
        factual: pd.DataFrame,
        schema: FeatureSchema,
        gt_generator,
        ground_truth_scale: str = "far",
        use_preferences: bool = False,
        noise_prob: float = 0.0,
        max_attempts: int = 64,
        seed: int | None = None,
    ) -> None:
        scale = str(ground_truth_scale).lower()
        if scale not in _GROUND_TRUTH_RADIUS:
            raise ValueError(
                "ground_truth_scale must be one of {'near', 'intermediate', 'far'}"
            )

        self._raw_factual = factual.copy(deep=True)
        self._factual = factual.loc[:, schema.feature_names].copy(deep=True)
        self._schema = schema
        self._eps = float(_GROUND_TRUTH_RADIUS[scale])
        self._catp = -1.0
        self._num_samples = 300
        self._use_preferences = bool(use_preferences)
        self._noise_prob = float(noise_prob)
        self._seed = seed
        self._ground_truth = self._generate_ground_truth_counterfactual(
            gt_generator=gt_generator,
            max_attempts=max_attempts,
        )
        self._preferences = sample_preference_weights(len(schema.feature_names))

    @property
    def factual(self) -> pd.DataFrame:
        return self._factual.copy(deep=True)

    @property
    def ground_truth(self) -> pd.DataFrame:
        return self._ground_truth.copy(deep=True)

    @property
    def preferences(self) -> np.ndarray:
        return self._preferences.copy()

    @property
    def raw_factual(self) -> pd.DataFrame:
        return self._raw_factual.copy(deep=True)

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def catp(self) -> float:
        return self._catp

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def _generate_ground_truth_counterfactual(
        self,
        gt_generator,
        max_attempts: int,
    ) -> pd.DataFrame:
        factual_values = self._factual.to_numpy(dtype="float32").reshape(-1)
        categorical_indices = list(self._schema.binary_indices)
        continuous_indices = list(self._schema.continuous_indices)
        immutable_indices = list(self._schema.immutable_indices)
        mutable_continuous = sorted(
            list(set(continuous_indices) - set(immutable_indices))
        )
        mutable_categorical = sorted(
            list(set(categorical_indices) - set(immutable_indices))
        )
        base_seed = 0 if self._seed is None else int(self._seed)

        with seed_context(base_seed):
            for attempt_index in range(max(1, int(max_attempts))):
                perturbed = factual_values.copy()
                if mutable_continuous:
                    direction = 2.0 * np.random.rand(len(mutable_continuous)) - 1.0
                    norm = np.linalg.norm(direction, ord=2)
                    if norm > 0.0:
                        direction = direction / norm * self._eps
                        perturbed[mutable_continuous] += direction.astype(np.float32)
                perturbed.clip(0.0, 1.0, out=perturbed)

                if mutable_categorical:
                    flip_mask = (
                        np.random.rand(len(mutable_categorical)) > 1.0 - self._catp
                    ).astype(float)
                    perturbed[mutable_categorical] = np.logical_xor(
                        perturbed[mutable_categorical],
                        flip_mask,
                    ).astype(np.float32)

                perturbed = project_actionability_array(
                    perturbed,
                    factual_values,
                    self._schema,
                )
                perturbed_df = pd.DataFrame(
                    perturbed.reshape(1, -1),
                    columns=self._schema.feature_names,
                )
                generator_seed = base_seed + int(attempt_index)
                with seed_context(generator_seed):
                    candidate = gt_generator.generate(perturbed_df)
                if not candidate.isna().any(axis=1).iloc[0]:
                    candidate = candidate.clip(lower=0.0, upper=1.0)
                    return candidate.reset_index(drop=True)

        fallback_seed = base_seed + int(max_attempts)
        with seed_context(fallback_seed):
            fallback = gt_generator.generate(self._factual)
        if not fallback.isna().any(axis=1).iloc[0]:
            fallback = fallback.clip(lower=0.0, upper=1.0)
            return fallback.reset_index(drop=True)
        return self._factual.reset_index(drop=True).copy(deep=True)

    def _score(self, candidate: pd.DataFrame) -> float:
        if candidate.isna().any(axis=1).iloc[0]:
            return float("inf")

        candidate_values = candidate.to_numpy(dtype="float32").reshape(-1)
        factual_values = self._factual.to_numpy(dtype="float32").reshape(-1)
        if self._use_preferences:
            return float(
                np.dot(self._preferences, np.abs(candidate_values - factual_values))
            )

        gt_values = self._ground_truth.to_numpy(dtype="float32").reshape(-1)
        return float(np.linalg.norm(candidate_values - gt_values, ord=2))

    def compare(self, candidate_a: pd.DataFrame, candidate_b: pd.DataFrame) -> int:
        score_a = self._score(candidate_a)
        score_b = self._score(candidate_b)
        prefer_second = not (score_a < score_b)

        if self._noise_prob > 0.0:
            if np.random.choice([0, 1], p=[1.0 - self._noise_prob, self._noise_prob]):
                prefer_second = not prefer_second
        elif self._noise_prob < 0.0:
            distance = np.linalg.norm(
                candidate_a.to_numpy(dtype="float32").reshape(-1)
                - candidate_b.to_numpy(dtype="float32").reshape(-1),
                ord=2,
            )

            def _sigmoid(alpha: float, beta: float, x: float) -> float:
                return 1.0 / (1.0 + np.exp(-alpha * (x + beta)))

            flip_probability = 1.0 - _sigmoid(2.0, 1.0, float(distance))
            if np.random.choice([0, 1], p=[1.0 - flip_probability, flip_probability]):
                prefer_second = not prefer_second

        return int(prefer_second)


def select_best_candidate(user: SimulatedUser, candidates: pd.DataFrame) -> int:
    if candidates.shape[0] == 0:
        raise ValueError("candidates must contain at least one row")
    if candidates.shape[0] == 1:
        return 0

    best_index = user.compare(candidates.iloc[[0]], candidates.iloc[[1]])
    for index in range(2, candidates.shape[0]):
        choose_new = user.compare(candidates.iloc[[best_index]], candidates.iloc[[index]])
        best_index = (1 - choose_new) * best_index + choose_new * index
    return int(best_index)
