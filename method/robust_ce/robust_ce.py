from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from dataset.dataset_object import DatasetObject
from method.method_object import MethodObject
from method.robust_ce.support import (
    ensure_solver_available,
    extract_mlp_parameters,
    normalize_objective_norm,
    normalize_solver_name,
    normalize_uncertainty_norm,
    resolve_feature_constraints,
    solve_robust_counterfactual,
)
from method.wachter.support import ensure_supported_target_model, validate_counterfactuals
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("robust_ce")
class RobustCeMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        rho: float = 0.1,
        uncertainty_norm: str = "l2",
        objective_norm: str = "l1",
        iterative: bool = True,
        time_limit: float | None = 100.0,
        max_iterations: int = 25,
        solver_name: str = "gurobi",
        solver_tee: bool = False,
        violation_tolerance: float = 1e-6,
        big_m_lower: float = -100.0,
        big_m_upper: float = 100.0,
        show_progress: bool = False,
        **kwargs,
    ):
        ensure_supported_target_model(
            target_model,
            (MlpModel,),
            "RobustCeMethod",
        )
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._rho = float(rho)
        self._uncertainty_norm = normalize_uncertainty_norm(uncertainty_norm)
        self._objective_norm = normalize_objective_norm(objective_norm)
        self._iterative = bool(iterative)
        self._time_limit = None if time_limit is None else float(time_limit)
        self._max_iterations = int(max_iterations)
        self._solver_name = normalize_solver_name(solver_name)
        self._solver_tee = bool(solver_tee)
        self._violation_tolerance = float(violation_tolerance)
        self._big_m_lower = float(big_m_lower)
        self._big_m_upper = float(big_m_upper)
        self._show_progress = bool(show_progress)
        self._last_run_stats: list[dict[str, object]] = []

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._rho <= 0:
            raise ValueError("rho must be > 0")
        if not self._iterative:
            raise ValueError("RobustCeMethod with MlpModel requires iterative=True")
        if self._time_limit is not None and self._time_limit <= 0:
            raise ValueError("time_limit must be > 0 when provided")
        if self._max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self._violation_tolerance < 0:
            raise ValueError("violation_tolerance must be >= 0")
        if self._big_m_lower >= 0:
            raise ValueError("big_m_lower must be < 0")
        if self._big_m_upper <= 0:
            raise ValueError("big_m_upper must be > 0")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for RobustCeMethod.fit()")

        with seed_context(self._seed):
            class_to_index = self._target_model.get_class_to_index()
            if len(class_to_index) != 2:
                raise ValueError(
                    "RobustCeMethod currently supports binary classification only"
                )
            if self._desired_class is not None and self._desired_class not in class_to_index:
                raise ValueError(
                    "desired_class is invalid for the trained target model"
                )

            ensure_solver_available(self._solver_name)
            self._feature_constraints = resolve_feature_constraints(trainset)
            self._feature_names = list(self._feature_constraints.feature_names)
            self._mlp_params = extract_mlp_parameters(self._target_model)
            if self._mlp_params.input_dim != len(self._feature_names):
                raise ValueError(
                    "Target MLP input dimension does not match the training feature width"
                )
            self._class_to_index = class_to_index
            self._last_run_stats = []
            self._is_trained = True

    def _predict_label_indices(self, factuals: pd.DataFrame) -> np.ndarray:
        prediction = self._target_model.get_prediction(factuals, proba=False)
        return prediction.detach().cpu().numpy().argmax(axis=1)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=1).any():
            raise ValueError("factuals must not contain NaN values")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        if factuals.empty:
            return factuals.copy(deep=True)

        original_prediction = self._predict_label_indices(factuals)
        rows: list[pd.Series] = []
        run_stats: list[dict[str, object]] = []

        with seed_context(self._seed):
            iterator = factuals.iterrows()
            if self._show_progress:
                iterator = tqdm(
                    iterator,
                    total=factuals.shape[0],
                    desc="robust_ce-rows",
                    leave=False,
                )

            for row_index, (_, row) in enumerate(iterator):
                if self._desired_class is None:
                    target_class_index = 1 - int(original_prediction[row_index])
                else:
                    target_class_index = int(
                        self._class_to_index[self._desired_class]
                    )
                    if int(original_prediction[row_index]) == target_class_index:
                        rows.append(
                            pd.Series(
                                np.nan,
                                index=self._feature_names,
                                dtype="float64",
                            )
                        )
                        run_stats.append(
                            {
                                "status": "already_desired_class",
                                "num_iterations": 0,
                                "comp_time": 0.0,
                                "master_trace": [],
                                "adversarial_objectives": [],
                            }
                        )
                        continue

                counterfactual, counterfactual_stats = solve_robust_counterfactual(
                    factual=row.astype("float64"),
                    feature_constraints=self._feature_constraints,
                    mlp_params=self._mlp_params,
                    target_class_index=target_class_index,
                    rho=self._rho,
                    uncertainty_norm=self._uncertainty_norm,
                    objective_norm=self._objective_norm,
                    solver_name=self._solver_name,
                    solver_tee=self._solver_tee,
                    time_limit=self._time_limit,
                    max_iterations=self._max_iterations,
                    violation_tolerance=self._violation_tolerance,
                    big_m_lower=self._big_m_lower,
                    big_m_upper=self._big_m_upper,
                )
                run_stats.append(counterfactual_stats)
                if counterfactual is None:
                    rows.append(
                        pd.Series(
                            np.nan,
                            index=self._feature_names,
                            dtype="float64",
                        )
                    )
                else:
                    rows.append(counterfactual.astype("float64"))

        self._last_run_stats = run_stats
        candidates = pd.DataFrame(
            rows,
            index=factuals.index,
            columns=self._feature_names,
        )
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
