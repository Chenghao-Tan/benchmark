from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import torch
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverStatus, TerminationCondition

from dataset.dataset_object import DatasetObject
from model.mlp.mlp import MlpModel
from preprocess.preprocess_utils import dataset_has_attr, resolve_feature_metadata

SUPPORTED_UNCERTAINTY_NORMS = {"l2", "linf"}
SUPPORTED_OBJECTIVE_NORMS = {"l1", "l2", "linf"}
SUPPORTED_SOLVERS = {"gurobi"}


@dataclass(frozen=True)
class OneHotGroup:
    source_feature: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class FeatureConstraints:
    feature_names: tuple[str, ...]
    binary_features: tuple[str, ...]
    integer_features: tuple[str, ...]
    onehot_groups: tuple[OneHotGroup, ...]
    immutable_features: tuple[str, ...]
    increase_only_features: tuple[str, ...]
    decrease_only_features: tuple[str, ...]
    nonnegative_features: tuple[str, ...]


@dataclass(frozen=True)
class MlpParameters:
    hidden_weights: tuple[np.ndarray, ...]
    hidden_biases: tuple[np.ndarray, ...]
    output_weights: np.ndarray
    output_bias: float
    input_dim: int


@dataclass
class MasterTraceStep:
    candidate: pd.Series
    objective_value: float | None


class RobustCeSolverError(RuntimeError):
    pass


def normalize_uncertainty_norm(uncertainty_norm: str) -> str:
    value = str(uncertainty_norm).lower()
    if value not in SUPPORTED_UNCERTAINTY_NORMS:
        raise ValueError(
            f"uncertainty_norm must be one of {sorted(SUPPORTED_UNCERTAINTY_NORMS)}"
        )
    return value


def normalize_objective_norm(objective_norm: str) -> str:
    value = str(objective_norm).lower()
    if value not in SUPPORTED_OBJECTIVE_NORMS:
        raise ValueError(
            f"objective_norm must be one of {sorted(SUPPORTED_OBJECTIVE_NORMS)}"
        )
    return value


def normalize_solver_name(solver_name: str) -> str:
    value = str(solver_name).lower()
    if value not in SUPPORTED_SOLVERS:
        raise ValueError(f"solver_name must be one of {sorted(SUPPORTED_SOLVERS)}")
    return value


def _infer_onehot_groups(
    feature_names: list[str],
    encoding_map: dict[str, list[str]] | None,
) -> tuple[OneHotGroup, ...]:
    if not encoding_map:
        return tuple()

    feature_name_set = set(feature_names)
    groups: list[OneHotGroup] = []
    for source_feature, encoded_columns in encoding_map.items():
        present_columns = [
            column for column in encoded_columns if column in feature_name_set
        ]
        if len(present_columns) < 2:
            continue
        if any("_therm_" in column for column in present_columns):
            raise NotImplementedError(
                "RobustCeMethod does not support thermometer-encoded categorical groups"
            )
        groups.append(
            OneHotGroup(
                source_feature=str(source_feature),
                columns=tuple(present_columns),
            )
        )
    return tuple(groups)


def _infer_integer_features(
    feature_df: pd.DataFrame,
    feature_names: list[str],
    feature_type: dict[str, str],
    binary_features: set[str],
    scaling_map: dict[str, str] | None,
) -> tuple[str, ...]:
    scaling_map = scaling_map or {}
    integer_features: list[str] = []
    for feature_name in feature_names:
        if feature_name in binary_features:
            continue
        if feature_name in scaling_map:
            continue
        if str(feature_type[feature_name]).lower() != "numerical":
            continue
        values = feature_df[feature_name].to_numpy(dtype=np.float64, copy=False)
        if values.size == 0:
            continue
        if np.all(np.isfinite(values)) and np.allclose(
            values, np.rint(values), atol=1e-8
        ):
            integer_features.append(feature_name)
    return tuple(integer_features)


def resolve_feature_constraints(trainset: DatasetObject) -> FeatureConstraints:
    feature_df = trainset.get(target=False)
    feature_names = list(feature_df.columns)
    feature_type, feature_mutability, feature_actionability = resolve_feature_metadata(
        trainset
    )

    encoding_map = (
        trainset.attr("encoding") if dataset_has_attr(trainset, "encoding") else None
    )
    onehot_groups = _infer_onehot_groups(feature_names, encoding_map)

    binary_features = tuple(
        feature_name
        for feature_name in feature_names
        if str(feature_type[feature_name]).lower() == "binary"
    )
    scaling_map = (
        trainset.attr("scaling") if dataset_has_attr(trainset, "scaling") else None
    )
    integer_features = _infer_integer_features(
        feature_df=feature_df,
        feature_names=feature_names,
        feature_type=feature_type,
        binary_features=set(binary_features),
        scaling_map=scaling_map,
    )

    immutable_features: list[str] = []
    increase_only_features: list[str] = []
    decrease_only_features: list[str] = []
    nonnegative_features: list[str] = []

    for feature_name in feature_names:
        actionability = str(feature_actionability[feature_name]).lower()
        is_mutable = bool(feature_mutability[feature_name])
        if (not is_mutable) or actionability in {"none", "same"}:
            immutable_features.append(feature_name)
        elif "increase" in actionability and "decrease" not in actionability:
            increase_only_features.append(feature_name)
        elif "decrease" in actionability and "increase" not in actionability:
            decrease_only_features.append(feature_name)

        values = feature_df[feature_name].to_numpy(dtype=np.float64, copy=False)
        if values.size and np.nanmin(values) >= -1e-9:
            nonnegative_features.append(feature_name)

    return FeatureConstraints(
        feature_names=tuple(feature_names),
        binary_features=binary_features,
        integer_features=integer_features,
        onehot_groups=onehot_groups,
        immutable_features=tuple(immutable_features),
        increase_only_features=tuple(increase_only_features),
        decrease_only_features=tuple(decrease_only_features),
        nonnegative_features=tuple(nonnegative_features),
    )


def extract_mlp_parameters(target_model: MlpModel) -> MlpParameters:
    model = getattr(target_model, "_model", None)
    if model is None:
        raise RuntimeError("Target model has not been initialized")

    linear_layers = [
        module for module in model.modules() if isinstance(module, torch.nn.Linear)
    ]
    if not linear_layers:
        raise ValueError("Target MLP does not expose any Linear layers")

    hidden_weights = tuple(
        np.asarray(layer.weight.detach().cpu().numpy(), dtype=np.float64)
        for layer in linear_layers[:-1]
    )
    hidden_biases = tuple(
        np.asarray(layer.bias.detach().cpu().numpy(), dtype=np.float64)
        for layer in linear_layers[:-1]
    )

    final_layer = linear_layers[-1]
    final_weight = np.asarray(
        final_layer.weight.detach().cpu().numpy(),
        dtype=np.float64,
    )
    final_bias = np.asarray(
        final_layer.bias.detach().cpu().numpy(),
        dtype=np.float64,
    )

    output_activation = str(
        getattr(target_model, "_output_activation_name", "softmax")
    ).lower()
    class_to_index = target_model.get_class_to_index()
    if len(class_to_index) != 2:
        raise ValueError("RobustCeMethod currently supports binary MlpModel only")

    if output_activation == "sigmoid":
        if final_weight.shape[0] != 1:
            raise ValueError("Sigmoid MlpModel must expose a single output logit")
        output_weights = final_weight[0].copy()
        output_bias = float(final_bias[0])
    elif output_activation == "softmax":
        if final_weight.shape[0] != 2:
            raise ValueError(
                "Softmax MlpModel must expose exactly two output logits for binary classification"
            )
        output_weights = final_weight[1] - final_weight[0]
        output_bias = float(final_bias[1] - final_bias[0])
    else:
        raise ValueError(
            "RobustCeMethod supports MlpModel output_activation 'sigmoid' or 'softmax' only"
        )

    input_dim = (
        int(hidden_weights[0].shape[1])
        if hidden_weights
        else int(output_weights.shape[0])
    )
    return MlpParameters(
        hidden_weights=hidden_weights,
        hidden_biases=hidden_biases,
        output_weights=np.asarray(output_weights, dtype=np.float64),
        output_bias=float(output_bias),
        input_dim=input_dim,
    )


def ensure_solver_available(solver_name: str) -> None:
    solver_name = normalize_solver_name(solver_name)
    solver = pyo.SolverFactory(solver_name)
    if not solver.available(exception_flag=False):
        raise RobustCeSolverError(
            f"Gurobi access failed: Pyomo solver '{solver_name}' is unavailable"
        )
    license_is_valid = getattr(solver, "license_is_valid", None)
    if callable(license_is_valid) and not bool(license_is_valid()):
        raise RobustCeSolverError(
            "Gurobi access failed: the configured Gurobi license is not usable in this environment"
        )


def _configure_tempdir() -> Path:
    candidates: list[Path] = []
    for env_var in ("TMPDIR", "TEMP", "TMP"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))
    candidates.extend([Path("/tmp"), Path("/var/tmp"), Path.cwd()])

    tempdir: Path | None = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            tempdir = candidate
            break

    if tempdir is None:
        try:
            tempdir = Path(tempfile.gettempdir())
        except Exception as exc:  # pragma: no cover
            raise RobustCeSolverError(
                "Gurobi access failed: no usable temporary directory is available for Pyomo"
            ) from exc

    tempdir_str = tempdir.as_posix()
    os.environ["TMPDIR"] = tempdir_str
    os.environ["TEMP"] = tempdir_str
    os.environ["TMP"] = tempdir_str
    TempfileManager.tempdir = tempdir_str
    return tempdir


def _build_base_model(
    factual: pd.Series,
    feature_constraints: FeatureConstraints,
) -> pyo.ConcreteModel:
    binary_features = set(feature_constraints.binary_features)
    integer_features = set(feature_constraints.integer_features)
    nonnegative_features = set(feature_constraints.nonnegative_features)

    def domain_rule(_model: pyo.ConcreteModel, feature_name: str):
        if feature_name in binary_features:
            return pyo.Binary
        if feature_name in integer_features:
            if feature_name in nonnegative_features:
                return pyo.NonNegativeIntegers
            return pyo.Integers
        return pyo.Reals

    model = pyo.ConcreteModel("RobustCE")
    model.x = pyo.Var(feature_constraints.feature_names, domain=domain_rule)
    model.base_constraints = pyo.ConstraintList()

    for group in feature_constraints.onehot_groups:
        model.base_constraints.add(
            pyo.quicksum(model.x[column] for column in group.columns) == 1.0
        )
    for feature_name in feature_constraints.immutable_features:
        model.base_constraints.add(model.x[feature_name] == float(factual[feature_name]))
    for feature_name in feature_constraints.increase_only_features:
        model.base_constraints.add(model.x[feature_name] >= float(factual[feature_name]))
    for feature_name in feature_constraints.decrease_only_features:
        model.base_constraints.add(model.x[feature_name] <= float(factual[feature_name]))
    for feature_name in feature_constraints.nonnegative_features:
        model.base_constraints.add(model.x[feature_name] >= 0.0)
    return model


def _add_distance_objective(
    model: pyo.ConcreteModel,
    factual: pd.Series,
    feature_names: tuple[str, ...],
    objective_norm: str,
) -> None:
    if objective_norm == "l2":
        model.OBJ = pyo.Objective(
            expr=pyo.quicksum(
                (model.x[feature_name] - float(factual[feature_name])) ** 2
                for feature_name in feature_names
            ),
            sense=pyo.minimize,
        )
        return

    if objective_norm == "linf":
        model.distance_aux = pyo.Var(domain=pyo.NonNegativeReals)
        model.distance_constraints = pyo.ConstraintList()
        for feature_name in feature_names:
            delta = model.x[feature_name] - float(factual[feature_name])
            model.distance_constraints.add(model.distance_aux >= delta)
            model.distance_constraints.add(model.distance_aux >= -delta)
        model.OBJ = pyo.Objective(expr=model.distance_aux, sense=pyo.minimize)
        return

    if objective_norm == "l1":
        model.distance_aux = pyo.Var(feature_names, domain=pyo.NonNegativeReals)
        model.distance_constraints = pyo.ConstraintList()
        for feature_name in feature_names:
            delta = model.x[feature_name] - float(factual[feature_name])
            model.distance_constraints.add(model.distance_aux[feature_name] >= delta)
            model.distance_constraints.add(model.distance_aux[feature_name] >= -delta)
        model.OBJ = pyo.Objective(
            expr=pyo.quicksum(
                model.distance_aux[feature_name] for feature_name in feature_names
            ),
            sense=pyo.minimize,
        )
        return

    raise ValueError(f"Unsupported objective norm: {objective_norm}")


def _add_class_constraints(
    constraint_list: pyo.ConstraintList,
    score_expr,
    target_class_index: int,
) -> None:
    if int(target_class_index) == 1:
        constraint_list.add(score_expr >= 0.0)
    elif int(target_class_index) == 0:
        constraint_list.add(score_expr <= 0.0)
    else:
        raise ValueError("Binary target_class_index must be 0 or 1")


def _output_expr_from_master_scenario(
    model: pyo.ConcreteModel,
    mlp_params: MlpParameters,
    feature_names: tuple[str, ...],
    scenario: dict[str, float],
    scenario_index: int,
    layer_index: int,
    node_index: int,
):
    if layer_index == 0:
        weights = mlp_params.hidden_weights[0]
        bias = mlp_params.hidden_biases[0][node_index]
        return pyo.quicksum(
            (model.x[feature_name] + float(scenario[feature_name]))
            * weights[node_index, input_index]
            for input_index, feature_name in enumerate(feature_names)
        ) + float(bias)

    weights = mlp_params.hidden_weights[layer_index]
    bias = mlp_params.hidden_biases[layer_index][node_index]
    prev_width = mlp_params.hidden_weights[layer_index - 1].shape[0]
    return pyo.quicksum(
        model.hidden_value[(layer_index - 1, prev_node_index, scenario_index)]
        * weights[node_index, prev_node_index]
        for prev_node_index in range(prev_width)
    ) + float(bias)


def _add_mlp_master_constraints(
    model: pyo.ConcreteModel,
    mlp_params: MlpParameters,
    feature_names: tuple[str, ...],
    scenarios: tuple[dict[str, float], ...],
    target_class_index: int,
    big_m_lower: float,
    big_m_upper: float,
) -> None:
    hidden_index = [
        (layer_index, node_index, scenario_index)
        for scenario_index in range(len(scenarios))
        for layer_index, weights in enumerate(mlp_params.hidden_weights)
        for node_index in range(weights.shape[0])
    ]
    if hidden_index:
        model.hidden_index = pyo.Set(initialize=hidden_index, dimen=3)
        model.hidden_value = pyo.Var(model.hidden_index, domain=pyo.NonNegativeReals)
        model.hidden_active = pyo.Var(model.hidden_index, domain=pyo.Binary)

    model.score_index = pyo.RangeSet(0, len(scenarios) - 1)
    model.score = pyo.Var(model.score_index, domain=pyo.Reals)
    model.nn_constraints = pyo.ConstraintList()

    for scenario_index, scenario in enumerate(scenarios):
        for layer_index, weights in enumerate(mlp_params.hidden_weights):
            for node_index in range(weights.shape[0]):
                affine_expr = _output_expr_from_master_scenario(
                    model=model,
                    mlp_params=mlp_params,
                    feature_names=feature_names,
                    scenario=scenario,
                    scenario_index=scenario_index,
                    layer_index=layer_index,
                    node_index=node_index,
                )
                hidden_key = (layer_index, node_index, scenario_index)
                model.nn_constraints.add(model.hidden_value[hidden_key] >= affine_expr)
                model.nn_constraints.add(
                    model.hidden_value[hidden_key]
                    <= big_m_upper * model.hidden_active[hidden_key]
                )
                model.nn_constraints.add(
                    model.hidden_value[hidden_key]
                    <= affine_expr - big_m_lower * (1 - model.hidden_active[hidden_key])
                )

        if mlp_params.hidden_weights:
            last_width = mlp_params.hidden_weights[-1].shape[0]
            output_expr = pyo.quicksum(
                model.hidden_value[
                    (len(mlp_params.hidden_weights) - 1, node_index, scenario_index)
                ]
                * mlp_params.output_weights[node_index]
                for node_index in range(last_width)
            ) + float(mlp_params.output_bias)
        else:
            output_expr = pyo.quicksum(
                model.x[feature_name] * mlp_params.output_weights[input_index]
                for input_index, feature_name in enumerate(feature_names)
            ) + float(mlp_params.output_bias)

        model.nn_constraints.add(model.score[scenario_index] == output_expr)
        _add_class_constraints(
            constraint_list=model.nn_constraints,
            score_expr=model.score[scenario_index],
            target_class_index=target_class_index,
        )


def _output_expr_from_adversarial_model(
    model: pyo.ConcreteModel,
    mlp_params: MlpParameters,
    feature_names: tuple[str, ...],
    layer_index: int,
    node_index: int,
):
    if layer_index == 0:
        weights = mlp_params.hidden_weights[0]
        bias = mlp_params.hidden_biases[0][node_index]
        return pyo.quicksum(
            model.x[feature_name] * weights[node_index, input_index]
            for input_index, feature_name in enumerate(feature_names)
        ) + float(bias)

    weights = mlp_params.hidden_weights[layer_index]
    bias = mlp_params.hidden_biases[layer_index][node_index]
    prev_width = mlp_params.hidden_weights[layer_index - 1].shape[0]
    return pyo.quicksum(
        model.hidden_value[(layer_index - 1, prev_node_index)]
        * weights[node_index, prev_node_index]
        for prev_node_index in range(prev_width)
    ) + float(bias)


def _add_mlp_adversarial_constraints(
    model: pyo.ConcreteModel,
    mlp_params: MlpParameters,
    feature_names: tuple[str, ...],
    target_class_index: int,
    big_m_lower: float,
    big_m_upper: float,
) -> None:
    hidden_index = [
        (layer_index, node_index)
        for layer_index, weights in enumerate(mlp_params.hidden_weights)
        for node_index in range(weights.shape[0])
    ]
    if hidden_index:
        model.hidden_index = pyo.Set(initialize=hidden_index, dimen=2)
        model.hidden_value = pyo.Var(model.hidden_index, domain=pyo.NonNegativeReals)
        model.hidden_active = pyo.Var(model.hidden_index, domain=pyo.Binary)

    model.score = pyo.Var(domain=pyo.Reals)
    model.nn_constraints = pyo.ConstraintList()

    for layer_index, weights in enumerate(mlp_params.hidden_weights):
        for node_index in range(weights.shape[0]):
            affine_expr = _output_expr_from_adversarial_model(
                model=model,
                mlp_params=mlp_params,
                feature_names=feature_names,
                layer_index=layer_index,
                node_index=node_index,
            )
            hidden_key = (layer_index, node_index)
            model.nn_constraints.add(model.hidden_value[hidden_key] >= affine_expr)
            model.nn_constraints.add(
                model.hidden_value[hidden_key]
                <= big_m_upper * model.hidden_active[hidden_key]
            )
            model.nn_constraints.add(
                model.hidden_value[hidden_key]
                <= affine_expr - big_m_lower * (1 - model.hidden_active[hidden_key])
            )

    if mlp_params.hidden_weights:
        last_width = mlp_params.hidden_weights[-1].shape[0]
        output_expr = pyo.quicksum(
            model.hidden_value[(len(mlp_params.hidden_weights) - 1, node_index)]
            * mlp_params.output_weights[node_index]
            for node_index in range(last_width)
        ) + float(mlp_params.output_bias)
    else:
        output_expr = pyo.quicksum(
            model.x[feature_name] * mlp_params.output_weights[input_index]
            for input_index, feature_name in enumerate(feature_names)
        ) + float(mlp_params.output_bias)

    model.nn_constraints.add(model.score == output_expr)
    _add_class_constraints(
        constraint_list=model.nn_constraints,
        score_expr=model.score,
        target_class_index=target_class_index,
    )


def _add_uncertainty_set_constraints(
    model: pyo.ConcreteModel,
    feature_names: tuple[str, ...],
    center: pd.Series,
    uncertainty_norm: str,
    rho: float,
) -> None:
    model.uncertainty_constraints = pyo.ConstraintList()
    if uncertainty_norm == "l2":
        model.uncertainty_constraints.add(
            pyo.quicksum(
                (model.x[feature_name] - float(center[feature_name])) ** 2
                for feature_name in feature_names
            )
            <= float(rho) ** 2
        )
        return

    if uncertainty_norm == "linf":
        for feature_name in feature_names:
            center_value = float(center[feature_name])
            model.uncertainty_constraints.add(
                model.x[feature_name] <= center_value + float(rho)
            )
            model.uncertainty_constraints.add(
                model.x[feature_name] >= center_value - float(rho)
            )
        return

    raise ValueError(f"Unsupported uncertainty norm: {uncertainty_norm}")


def _remaining_time(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    return max(0.0, remaining)


def _solve_model(
    model: pyo.ConcreteModel,
    solver_name: str,
    time_limit: float | None,
    solver_tee: bool,
):
    _configure_tempdir()
    solver = pyo.SolverFactory(solver_name)
    if time_limit is not None:
        solver.options["TimeLimit"] = float(time_limit)
    try:
        return solver.solve(model, tee=bool(solver_tee))
    except Exception as exc:  # pragma: no cover
        message = str(exc)
        lowered = message.lower()
        if any(
            pattern in lowered
            for pattern in [
                "could not resolve host",
                "token.gurobi.com",
                "license",
                "wls",
                "token server",
                "user name mismatch",
                "hostid mismatch",
            ]
        ):
            raise RobustCeSolverError(f"Gurobi access failed: {message}") from None
        raise


def _extract_feature_solution(
    model: pyo.ConcreteModel,
    feature_names: tuple[str, ...],
) -> pd.Series | None:
    values: list[float] = []
    for feature_name in feature_names:
        try:
            value = pyo.value(model.x[feature_name])
        except Exception:
            return None
        if value is None or not np.isfinite(value):
            return None
        values.append(float(value))
    return pd.Series(values, index=list(feature_names), dtype="float64")


def _extract_scalar_objective(model: pyo.ConcreteModel) -> float | None:
    try:
        objective_value = pyo.value(model.OBJ)
    except Exception:
        return None
    if objective_value is None or not np.isfinite(objective_value):
        return None
    return float(objective_value)


def _is_infeasible(results) -> bool:
    termination = results.solver.termination_condition
    return termination in {
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.invalidProblem,
    }


def _is_acceptable(results) -> bool:
    status = results.solver.status
    termination = results.solver.termination_condition
    return status in {SolverStatus.ok, SolverStatus.warning} and termination in {
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.globallyOptimal,
        TerminationCondition.maxTimeLimit,
    }


def _project_candidate(
    candidate: pd.Series,
    factual: pd.Series,
    feature_constraints: FeatureConstraints,
) -> pd.Series:
    projected = candidate.copy(deep=True).astype("float64")

    for feature_name in feature_constraints.binary_features:
        projected[feature_name] = 1.0 if projected[feature_name] >= 0.5 else 0.0

    for feature_name in feature_constraints.integer_features:
        projected[feature_name] = float(np.rint(projected[feature_name]))

    for group in feature_constraints.onehot_groups:
        group_columns = list(group.columns)
        values = projected.loc[group_columns].to_numpy(dtype=np.float64, copy=False)
        winner = int(np.nanargmax(values)) if values.size else 0
        projected.loc[group_columns] = 0.0
        projected[group_columns[winner]] = 1.0

    for feature_name in feature_constraints.nonnegative_features:
        if projected[feature_name] < 0.0 and abs(projected[feature_name]) <= 1e-6:
            projected[feature_name] = 0.0

    for feature_name in feature_constraints.immutable_features:
        projected[feature_name] = float(factual[feature_name])
    for feature_name in feature_constraints.increase_only_features:
        if projected[feature_name] < factual[feature_name] and (
            factual[feature_name] - projected[feature_name]
        ) <= 1e-6:
            projected[feature_name] = float(factual[feature_name])
    for feature_name in feature_constraints.decrease_only_features:
        if projected[feature_name] > factual[feature_name] and (
            projected[feature_name] - factual[feature_name]
        ) <= 1e-6:
            projected[feature_name] = float(factual[feature_name])

    return projected.reindex(list(feature_constraints.feature_names))


def _scenario_from_points(
    center: pd.Series,
    adversarial_point: pd.Series,
    feature_names: tuple[str, ...],
) -> dict[str, float]:
    return {
        feature_name: float(adversarial_point[feature_name] - center[feature_name])
        for feature_name in feature_names
    }


def _scenario_duplicate(
    scenario: dict[str, float],
    scenarios: tuple[dict[str, float], ...],
    feature_names: tuple[str, ...],
    tolerance: float = 1e-6,
) -> bool:
    for existing in scenarios:
        if all(
            abs(float(existing[feature_name]) - float(scenario[feature_name]))
            <= tolerance
            for feature_name in feature_names
        ):
            return True
    return False


def compute_distance_to_class(
    center: pd.Series,
    mlp_params: MlpParameters,
    target_class_index: int,
    objective_norm: str,
    solver_name: str,
    solver_tee: bool,
    time_limit: float | None,
    big_m_lower: float,
    big_m_upper: float,
) -> tuple[pd.Series | None, float | None]:
    feature_names = tuple(center.index)
    objective_norm = normalize_objective_norm(objective_norm)
    solver_name = normalize_solver_name(solver_name)

    model = pyo.ConcreteModel("RobustCEBorderDistance")
    model.x = pyo.Var(feature_names, domain=pyo.Reals)
    _add_distance_objective(
        model=model,
        factual=center,
        feature_names=feature_names,
        objective_norm=objective_norm,
    )
    _add_mlp_adversarial_constraints(
        model=model,
        mlp_params=mlp_params,
        feature_names=feature_names,
        target_class_index=target_class_index,
        big_m_lower=big_m_lower,
        big_m_upper=big_m_upper,
    )
    results = _solve_model(
        model=model,
        solver_name=solver_name,
        time_limit=time_limit,
        solver_tee=solver_tee,
    )
    if _is_infeasible(results) or not _is_acceptable(results):
        return None, None
    point = _extract_feature_solution(model, feature_names)
    distance = _extract_scalar_objective(model)
    if point is None or distance is None:
        return None, None
    if objective_norm == "l2":
        distance = float(np.sqrt(max(distance, 0.0)))
    return point, distance


def solve_robust_counterfactual(
    factual: pd.Series,
    feature_constraints: FeatureConstraints,
    mlp_params: MlpParameters,
    target_class_index: int,
    rho: float,
    uncertainty_norm: str,
    objective_norm: str,
    solver_name: str,
    solver_tee: bool,
    time_limit: float | None,
    max_iterations: int,
    violation_tolerance: float,
    big_m_lower: float,
    big_m_upper: float,
) -> tuple[pd.Series | None, dict[str, object]]:
    uncertainty_norm = normalize_uncertainty_norm(uncertainty_norm)
    objective_norm = normalize_objective_norm(objective_norm)
    solver_name = normalize_solver_name(solver_name)

    start_time = time.monotonic()
    deadline = None if time_limit is None else time.monotonic() + float(time_limit)
    feature_names = feature_constraints.feature_names
    scenarios: tuple[dict[str, float], ...] = (
        {feature_name: 0.0 for feature_name in feature_names},
    )
    adversarial_class_index = 1 - int(target_class_index)
    stats: dict[str, object] = {
        "status": "unknown",
        "num_iterations": 0,
        "comp_time": float("nan"),
        "master_trace": [],
        "adversarial_objectives": [],
    }

    def finalize(
        candidate: pd.Series | None,
        status: str,
    ) -> tuple[pd.Series | None, dict[str, object]]:
        stats["status"] = status
        stats["comp_time"] = float(time.monotonic() - start_time)
        return candidate, stats

    for _ in range(int(max_iterations)):
        remaining = _remaining_time(deadline)
        if remaining is not None and remaining <= 0.0:
            return finalize(None, "time_limit")

        master_model = _build_base_model(
            factual=factual,
            feature_constraints=feature_constraints,
        )
        _add_distance_objective(
            model=master_model,
            factual=factual,
            feature_names=feature_names,
            objective_norm=objective_norm,
        )
        _add_mlp_master_constraints(
            model=master_model,
            mlp_params=mlp_params,
            feature_names=feature_names,
            scenarios=scenarios,
            target_class_index=target_class_index,
            big_m_lower=big_m_lower,
            big_m_upper=big_m_upper,
        )
        master_results = _solve_model(
            model=master_model,
            solver_name=solver_name,
            time_limit=remaining,
            solver_tee=solver_tee,
        )
        if _is_infeasible(master_results):
            return finalize(None, "master_infeasible")
        if not _is_acceptable(master_results):
            return finalize(None, "master_unacceptable")

        candidate = _extract_feature_solution(master_model, feature_names)
        if candidate is None:
            return finalize(None, "master_no_solution")
        candidate = _project_candidate(
            candidate=candidate,
            factual=factual,
            feature_constraints=feature_constraints,
        )
        stats["master_trace"].append(
            MasterTraceStep(
                candidate=candidate.copy(deep=True),
                objective_value=_extract_scalar_objective(master_model),
            )
        )

        remaining = _remaining_time(deadline)
        if remaining is not None and remaining <= 0.0:
            return finalize(None, "time_limit")

        adversarial_model = _build_base_model(
            factual=factual,
            feature_constraints=feature_constraints,
        )
        _add_mlp_adversarial_constraints(
            model=adversarial_model,
            mlp_params=mlp_params,
            feature_names=feature_names,
            target_class_index=adversarial_class_index,
            big_m_lower=big_m_lower,
            big_m_upper=big_m_upper,
        )
        _add_uncertainty_set_constraints(
            model=adversarial_model,
            feature_names=feature_names,
            center=candidate,
            uncertainty_norm=uncertainty_norm,
            rho=rho,
        )
        if adversarial_class_index == 1:
            adversarial_model.OBJ = pyo.Objective(
                expr=adversarial_model.score,
                sense=pyo.maximize,
            )
        else:
            adversarial_model.OBJ = pyo.Objective(
                expr=-adversarial_model.score,
                sense=pyo.maximize,
            )

        adversarial_results = _solve_model(
            model=adversarial_model,
            solver_name=solver_name,
            time_limit=remaining,
            solver_tee=solver_tee,
        )
        if _is_infeasible(adversarial_results):
            return finalize(candidate, "adversarial_infeasible")
        if not _is_acceptable(adversarial_results):
            return finalize(None, "adversarial_unacceptable")

        adversarial_point = _extract_feature_solution(adversarial_model, feature_names)
        violation = _extract_scalar_objective(adversarial_model)
        if adversarial_point is None or violation is None:
            return finalize(None, "adversarial_no_solution")
        stats["adversarial_objectives"].append(float(violation))
        if violation <= float(violation_tolerance):
            return finalize(candidate, "success")

        scenario = _scenario_from_points(
            center=candidate,
            adversarial_point=adversarial_point,
            feature_names=feature_names,
        )
        if _scenario_duplicate(
            scenario=scenario,
            scenarios=scenarios,
            feature_names=feature_names,
        ):
            return finalize(None, "duplicate_scenario")
        scenarios = scenarios + (scenario,)
        stats["num_iterations"] = int(stats["num_iterations"]) + 1

    return finalize(None, "max_iterations")
