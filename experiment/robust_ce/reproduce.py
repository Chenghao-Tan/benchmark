from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("TEMP", "/tmp")
os.environ.setdefault("TMP", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from evaluation.distance import DistanceEvaluation
from evaluation.validity import ValidityEvaluation
from experiment import Experiment
from method.robust_ce.support import MasterTraceStep, compute_distance_to_class

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yml")


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _apply_device(config: dict, device: str) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["device"] = device
    cfg["method"]["device"] = device
    return cfg


def _materialize_datasets(experiment: Experiment) -> tuple[object, object]:
    datasets = [experiment._raw_dataset]
    for preprocess_step in experiment._preprocess:
        next_datasets = []
        for current_dataset in datasets:
            transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                next_datasets.append(transformed)
        datasets = next_datasets
    return experiment._resolve_train_test(datasets)


def _clone_with_df(dataset, df: pd.DataFrame, *, factual: bool) -> object:
    cloned = dataset.clone()
    flag = "testset" if factual else "counterfactual"
    cloned.update(flag, True, df=df.copy(deep=True))
    cloned.freeze()
    return cloned


def _make_factual_subset(dataset, num_instances: int):
    feature_df = dataset.get(target=False).iloc[:num_instances].copy(deep=True)
    target_df = dataset.get(target=True).iloc[:num_instances].copy(deep=True)
    subset_df = pd.concat([feature_df, target_df], axis=1)
    subset_df = subset_df.loc[:, dataset.ordered_features()]
    return _clone_with_df(dataset, subset_df, factual=True)


def _make_counterfactual_dataset(factual_dataset, counterfactual_features: pd.DataFrame):
    target_column = factual_dataset.target_column
    counterfactual_target = pd.DataFrame(
        -1.0,
        index=counterfactual_features.index,
        columns=[target_column],
    )
    counterfactual_df = pd.concat(
        [counterfactual_features.copy(deep=True), counterfactual_target],
        axis=1,
    )
    counterfactual_df = counterfactual_df.loc[:, factual_dataset.ordered_features()]
    return _clone_with_df(factual_dataset, counterfactual_df, factual=False)


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1) / np.sqrt(len(values)))


def _format_layer_variant(layers: list[int]) -> str:
    if len(layers) == 1:
        return f"({layers[0]},)"
    return "(" + ", ".join(str(layer) for layer in layers) + ")"


def _row_success_mask(counterfactual_features: pd.DataFrame) -> pd.Series:
    return ~counterfactual_features.isna().any(axis=1)


def _compute_border_distance_summary(
    method_obj,
    stats: list[dict[str, object]],
    original_prediction: np.ndarray,
    uncertainty_norm: str,
    border_time_limit: float | None,
    show_progress: bool = False,
) -> tuple[list[float], list[int], list[str]]:
    early_stop_distances: list[float] = []
    early_stop_iterations: list[int] = []
    failure_statuses: list[str] = []

    iterator = enumerate(stats)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(stats),
            desc="border-distance",
            leave=False,
        )

    for row_index, row_stats in iterator:
        if str(row_stats.get("status", "")).lower() in {"success", "adversarial_infeasible"}:
            continue

        failure_statuses.append(str(row_stats.get("status", "unknown")))
        best_distance = float("-inf")
        for trace_step in row_stats.get("master_trace", []):
            if not isinstance(trace_step, MasterTraceStep):
                continue
            _, distance = compute_distance_to_class(
                center=trace_step.candidate,
                mlp_params=method_obj._mlp_params,
                target_class_index=int(original_prediction[row_index]),
                objective_norm=uncertainty_norm,
                solver_name=method_obj._solver_name,
                solver_tee=method_obj._solver_tee,
                time_limit=border_time_limit,
                big_m_lower=method_obj._big_m_lower,
                big_m_upper=method_obj._big_m_upper,
            )
            if distance is None:
                continue
            best_distance = max(best_distance, float(distance))

        early_stop_distances.append(
            float(best_distance) if np.isfinite(best_distance) else float("nan")
        )
        early_stop_iterations.append(int(row_stats.get("num_iterations", 0)))

    return early_stop_distances, early_stop_iterations, failure_statuses


def _compute_secondary_metrics(factual_dataset, counterfactual_features: pd.DataFrame) -> dict[str, float]:
    counterfactual_dataset = _make_counterfactual_dataset(
        factual_dataset,
        counterfactual_features,
    )
    validity = ValidityEvaluation().evaluate(
        factual_dataset,
        counterfactual_dataset,
    ).iloc[0]["validity"]
    distance = DistanceEvaluation(metrics=["l1", "linf"]).evaluate(
        factual_dataset,
        counterfactual_dataset,
    ).iloc[0]
    return {
        "validity": float(validity),
        "distance_l1": float(distance["distance_l1"]),
        "distance_linf": float(distance["distance_linf"]),
    }


def _safe_nanmean(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def _print_results(device: str, results: list[dict[str, object]]) -> None:
    summary_rows = []
    diagnostics_rows = []
    for result in results:
        summary_rows.append(
            {
                "layers": result["layer_key"],
                "success": f"{result['num_success']}/{result['num_rows']}",
                "time_mean": result["comp_time_mean"],
                "time_sem": result["comp_time_sem"],
                "iter_mean": result["num_iterations_mean"],
                "iter_sem": result["num_iterations_sem"],
                "early_stop_count": result["early_stop_count"],
                "early_dist_mean": result["early_stop_distance_mean"],
                "early_iter_mean": result["early_stop_iteration_mean"],
            }
        )
        diagnostics_rows.append(
            {
                "layers": result["layer_key"],
                "validity": result["secondary_metrics"]["validity"],
                "distance_l1": result["secondary_metrics"]["distance_l1"],
                "distance_linf": result["secondary_metrics"]["distance_linf"],
                "failure_statuses": ", ".join(result["failure_statuses"])
                if result["failure_statuses"]
                else "",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)

    print(f"Device: {device}")
    print()
    print("Reproduction Summary")
    print(summary_df.to_string(index=False))
    print()
    print("Secondary Diagnostics")
    print(diagnostics_df.to_string(index=False))


def _run_variant(base_config: dict, layers: list[int], device: str) -> dict[str, object]:
    config = _apply_device(base_config, device)
    config["name"] = f"{config['name']}_{'_'.join(str(layer) for layer in layers)}"
    config["model"]["layers"] = list(layers)

    experiment = Experiment(config)
    trainset, testset = _materialize_datasets(experiment)

    experiment._target_model.fit(trainset)
    experiment._method.fit(trainset)

    num_instances = int(config["reproduction"]["num_instances"])
    factual_dataset = _make_factual_subset(testset, num_instances=num_instances)
    original_prediction = (
        experiment._target_model.predict(factual_dataset).argmax(dim=1).detach().cpu().numpy()
    )
    counterfactual_dataset = experiment._method.predict(
        factual_dataset,
        batch_size=num_instances,
    )
    counterfactual_features = counterfactual_dataset.get(target=False)
    success_mask = _row_success_mask(counterfactual_features)

    row_stats = list(getattr(experiment._method, "_last_run_stats", []))
    border_time_limit = config["reproduction"].get("border_distance_time_limit")
    early_stop_distances, early_stop_iterations, failure_statuses = _compute_border_distance_summary(
        experiment._method,
        row_stats,
        original_prediction,
        uncertainty_norm=str(config["method"]["uncertainty_norm"]).lower(),
        border_time_limit=None if border_time_limit is None else float(border_time_limit),
    )

    comp_time_list = [
        float(stat.get("comp_time", float("nan")))
        for stat, success in zip(row_stats, success_mask.tolist())
        if success and np.isfinite(float(stat.get("comp_time", float("nan"))))
    ]
    num_iterations_list = [
        int(stat.get("num_iterations", 0))
        for stat, success in zip(row_stats, success_mask.tolist())
        if success
    ]

    secondary_metrics = _compute_secondary_metrics(
        factual_dataset,
        counterfactual_features,
    )

    return {
        "layers": list(layers),
        "layer_key": _format_layer_variant(layers),
        "num_rows": int(len(counterfactual_features)),
        "num_success": int(success_mask.sum()),
        "comp_time_mean": float(np.mean(comp_time_list)) if comp_time_list else float("nan"),
        "comp_time_sem": _sem(comp_time_list),
        "num_iterations_mean": float(np.mean(num_iterations_list)) if num_iterations_list else float("nan"),
        "num_iterations_sem": _sem([float(value) for value in num_iterations_list]),
        "early_stop_count": int(len(early_stop_distances)),
        "early_stop_distance_mean": _safe_nanmean(early_stop_distances),
        "early_stop_distance_sem": _sem(
            [value for value in early_stop_distances if np.isfinite(value)]
        ),
        "early_stop_iteration_mean": float(np.mean(early_stop_iterations)) if early_stop_iterations else float("nan"),
        "early_stop_iteration_sem": _sem([float(value) for value in early_stop_iterations]),
        "failure_statuses": failure_statuses,
        "secondary_metrics": secondary_metrics,
    }


def _run_variant_with_progress(
    base_config: dict,
    layers: list[int],
    device: str,
    show_progress: bool,
) -> dict[str, object]:
    config = _apply_device(base_config, device)
    config["name"] = f"{config['name']}_{'_'.join(str(layer) for layer in layers)}"
    config["model"]["layers"] = list(layers)
    config["method"]["show_progress"] = bool(show_progress)

    experiment = Experiment(config)
    trainset, testset = _materialize_datasets(experiment)

    experiment._target_model.fit(trainset)
    experiment._method.fit(trainset)

    num_instances = int(config["reproduction"]["num_instances"])
    factual_dataset = _make_factual_subset(testset, num_instances=num_instances)
    original_prediction = (
        experiment._target_model.predict(factual_dataset).argmax(dim=1).detach().cpu().numpy()
    )
    counterfactual_dataset = experiment._method.predict(
        factual_dataset,
        batch_size=num_instances,
    )
    counterfactual_features = counterfactual_dataset.get(target=False)
    success_mask = _row_success_mask(counterfactual_features)

    row_stats = list(getattr(experiment._method, "_last_run_stats", []))
    border_time_limit = config["reproduction"].get("border_distance_time_limit")
    early_stop_distances, early_stop_iterations, failure_statuses = _compute_border_distance_summary(
        experiment._method,
        row_stats,
        original_prediction,
        uncertainty_norm=str(config["method"]["uncertainty_norm"]).lower(),
        border_time_limit=None if border_time_limit is None else float(border_time_limit),
        show_progress=show_progress,
    )

    comp_time_list = [
        float(stat.get("comp_time", float("nan")))
        for stat, success in zip(row_stats, success_mask.tolist())
        if success and np.isfinite(float(stat.get("comp_time", float("nan"))))
    ]
    num_iterations_list = [
        int(stat.get("num_iterations", 0))
        for stat, success in zip(row_stats, success_mask.tolist())
        if success
    ]

    secondary_metrics = _compute_secondary_metrics(
        factual_dataset,
        counterfactual_features,
    )

    return {
        "layers": list(layers),
        "layer_key": _format_layer_variant(layers),
        "num_rows": int(len(counterfactual_features)),
        "num_success": int(success_mask.sum()),
        "comp_time_mean": float(np.mean(comp_time_list)) if comp_time_list else float("nan"),
        "comp_time_sem": _sem(comp_time_list),
        "num_iterations_mean": float(np.mean(num_iterations_list)) if num_iterations_list else float("nan"),
        "num_iterations_sem": _sem([float(value) for value in num_iterations_list]),
        "early_stop_count": int(len(early_stop_distances)),
        "early_stop_distance_mean": _safe_nanmean(early_stop_distances),
        "early_stop_distance_sem": _sem(
            [value for value in early_stop_distances if np.isfinite(value)]
        ),
        "early_stop_iteration_mean": float(np.mean(early_stop_iterations)) if early_stop_iterations else float("nan"),
        "early_stop_iteration_sem": _sem([float(value) for value in early_stop_iterations]),
        "failure_statuses": failure_statuses,
        "secondary_metrics": secondary_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = _load_config(config_path)
    device = _resolve_device()

    reproduction_cfg = config.get("reproduction", {})
    hidden_layer_variants = [
        [int(layer) for layer in variant]
        for variant in reproduction_cfg.get("hidden_layer_variants", [])
    ]
    show_progress = bool(reproduction_cfg.get("show_progress", True))

    results = []
    iterator = hidden_layer_variants
    if show_progress:
        iterator = tqdm(hidden_layer_variants, desc="mlp-variants")

    for layers in iterator:
        result = _run_variant_with_progress(
            config,
            layers,
            device=device,
            show_progress=show_progress,
        )
        results.append(result)

    _print_results(device, results)


if __name__ == "__main__":
    main()
