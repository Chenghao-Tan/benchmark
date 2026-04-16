from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from preprocess.preprocess_object import PreProcessObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from experiment import Experiment
from method.mace.library import normalizedDistance
from dataset.dataset_object import DatasetObject
from utils.registry import register

DEFAULT_CONFIG_PATHS = {
    "adult": Path(__file__).with_name("adult_randomforest_mace_reproduce.yaml"),
    "credit": Path(__file__).with_name("credit_randomforest_mace_reproduce.yaml"),
    "compas": Path(__file__).with_name("compas_randomforest_mace_reproduce.yaml"),
}
LOCAL_DISTANCE_EVALUATION_NAME = "mace_normalized_distance_local"
LOCAL_SELECT_PREPROCESS_NAME = "mace_select_test_indices_local"
EPSILON_SEQUENCE = [1.0e-1, 1.0e-3, 1.0e-5]
NORM_TO_METRIC = {
    "zero_norm": "distance_l0",
    "one_norm": "distance_l1",
    "infty_norm": "distance_linf",
}
MACE_DISTANCE_NORMS = {
    "distance_l0": "zero_norm",
    "distance_l1": "one_norm",
    "distance_linf": "infty_norm",
}


@register(LOCAL_SELECT_PREPROCESS_NAME)
class MaceSelectTestIndicesPreProcess(PreProcessObject):
    def __init__(
        self,
        seed: int | None = None,
        indices: list[int] | list[str] | None = None,
        **kwargs,
    ):
        del kwargs
        self._seed = seed
        self._indices = [] if indices is None else list(indices)

    def transform(self, input):
        if not getattr(input, "testset", False):
            return input
        filtered_df = input.snapshot().loc[self._indices].copy(deep=True)
        input.update("selected_test_indices", list(self._indices), df=filtered_df)
        return input


def _dataset_has_attr(dataset, flag: str) -> bool:
    try:
        dataset.attr(flag)
    except AttributeError:
        return False
    return True


def _is_integer_valued(series: pd.Series) -> bool:
    values = series.dropna().to_numpy(dtype="float64")
    if values.size == 0:
        return False
    return bool(np.allclose(values, np.round(values)))


def _normalize_actionability(value: object) -> str:
    normalized = str(value).lower()
    if normalized == "same":
        return "none"
    if normalized not in {"none", "any", "same-or-increase", "same-or-decrease"}:
        raise ValueError(f"Unsupported MACE actionability: {value}")
    return normalized


class _LocalMaceAttribute:
    def __init__(
        self,
        *,
        attr_name_kurz: str,
        attr_type: str,
        lower_bound: float,
        upper_bound: float,
        mutability: bool,
        actionability: str,
    ):
        self.attr_name_kurz = attr_name_kurz
        self.attr_type = attr_type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mutability = mutability
        self.actionability = actionability


class _LocalMaceDatasetWrapper:
    def __init__(
        self,
        *,
        dataset_name: str,
        feature_names: list[str],
        feature_types: dict[str, str],
        bounds: dict[str, tuple[float, float]],
        mutability: dict[str, bool],
        actionability: dict[str, str],
    ):
        self.dataset_name = dataset_name
        self.is_one_hot = False
        self._feature_names = list(feature_names)
        self._feature_types = dict(feature_types)
        self._short_names = {
            feature_name: f"x{index}"
            for index, feature_name in enumerate(self._feature_names)
        }
        self._inverse_short_names = {
            short_name: feature_name
            for feature_name, short_name in self._short_names.items()
        }
        self.attributes_kurz: dict[str, _LocalMaceAttribute] = {}

        for feature_name in self._feature_names:
            short_name = self._short_names[feature_name]
            lower_bound, upper_bound = bounds[feature_name]
            self.attributes_kurz[short_name] = _LocalMaceAttribute(
                attr_name_kurz=short_name,
                attr_type=str(feature_types[feature_name]),
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                mutability=bool(mutability[feature_name]),
                actionability=str(actionability[feature_name]).lower(),
            )
        self.attributes_kurz["y"] = _LocalMaceAttribute(
            attr_name_kurz="y",
            attr_type="binary",
            lower_bound=0.0,
            upper_bound=1.0,
            mutability=False,
            actionability="none",
        )

    @classmethod
    def from_dataset(cls, dataset) -> "_LocalMaceDatasetWrapper":
        feature_df = dataset.get(target=False)
        feature_names = list(feature_df.columns)

        has_all_mace_metadata = all(
            _dataset_has_attr(dataset, key)
            for key in (
                "mace_feature_type",
                "mace_feature_mutability",
                "mace_feature_actionability",
            )
        )
        if has_all_mace_metadata:
            feature_type_raw = dataset.attr("mace_feature_type")
            feature_mutability_raw = dataset.attr("mace_feature_mutability")
            feature_actionability_raw = dataset.attr("mace_feature_actionability")
            feature_type = {
                feature_name: str(feature_type_raw[feature_name]).lower()
                for feature_name in feature_names
            }
        else:
            raw_feature_type = dataset.attr("raw_feature_type")
            feature_mutability_raw = dataset.attr("raw_feature_mutability")
            feature_actionability_raw = dataset.attr("raw_feature_actionability")
            feature_type = {}
            for feature_name in feature_names:
                raw_type = str(raw_feature_type[feature_name]).lower()
                if raw_type == "numerical":
                    feature_type[feature_name] = (
                        "numeric-int"
                        if _is_integer_valued(feature_df[feature_name])
                        else "numeric-real"
                    )
                elif raw_type in {"binary", "categorical"}:
                    feature_type[feature_name] = raw_type
                else:
                    raise ValueError(
                        f"Unsupported raw_feature_type for MACE fallback: {raw_type}"
                    )

        feature_min = None
        feature_max = None
        if _dataset_has_attr(dataset, "balanced"):
            balanced = dataset.attr("balanced")
            if isinstance(balanced, dict):
                raw_feature_min = balanced.get("feature_min")
                raw_feature_max = balanced.get("feature_max")
                if isinstance(raw_feature_min, dict) and isinstance(
                    raw_feature_max, dict
                ):
                    feature_min = {
                        feature_name: float(raw_feature_min[feature_name])
                        for feature_name in feature_names
                    }
                    feature_max = {
                        feature_name: float(raw_feature_max[feature_name])
                        for feature_name in feature_names
                    }

        bounds = {}
        for feature_name in feature_names:
            if feature_min is not None and feature_max is not None:
                bounds[feature_name] = (
                    float(feature_min[feature_name]),
                    float(feature_max[feature_name]),
                )
            else:
                bounds[feature_name] = (
                    float(feature_df[feature_name].min()),
                    float(feature_df[feature_name].max()),
                )

        feature_mutability = {
            feature_name: bool(feature_mutability_raw[feature_name])
            for feature_name in feature_names
        }
        feature_actionability = {
            feature_name: _normalize_actionability(
                feature_actionability_raw[feature_name]
            )
            for feature_name in feature_names
        }
        dataset_name = str(dataset.attr("name")) if _dataset_has_attr(dataset, "name") else ""
        return cls(
            dataset_name=dataset_name,
            feature_names=feature_names,
            feature_types=feature_type,
            bounds=bounds,
            mutability=feature_mutability,
            actionability=feature_actionability,
        )

    def getInputAttributeNames(self, kind: str = "kurz"):
        del kind
        return [self._short_names[feature_name] for feature_name in self._feature_names]

    def getOutputAttributeNames(self, kind: str = "kurz"):
        del kind
        return ["y"]

    def getInputOutputAttributeNames(self, kind: str = "kurz"):
        return self.getInputAttributeNames(kind) + self.getOutputAttributeNames(kind)

    def getMutableAttributeNames(self, kind: str = "kurz"):
        del kind
        return [
            self._short_names[feature_name]
            for feature_name in self._feature_names
            if self.attributes_kurz[self._short_names[feature_name]].mutability
        ]

    def getOneHotAttributesNames(self, kind: str = "kurz"):
        del kind
        return []

    def getNonHotAttributesNames(self, kind: str = "kurz"):
        del kind
        return self.getInputAttributeNames()

    def getSiblingsFor(self, attr_name_kurz: str):
        return [attr_name_kurz]

    def feature_row_to_short_dict(
        self,
        row: pd.Series,
        predicted_label: int | None = None,
    ) -> dict[str, int | float | bool]:
        output: dict[str, int | float | bool] = {}
        for feature_name in self._feature_names:
            value = row[feature_name]
            if self._feature_types[feature_name] == "numeric-real":
                output[self._short_names[feature_name]] = float(value)
            else:
                output[self._short_names[feature_name]] = int(round(float(value)))
        output["y"] = bool(predicted_label) if predicted_label is not None else False
        return output

    def short_dict_to_feature_row(self, sample: dict[str, object]) -> pd.Series:
        row = {}
        for short_name, feature_name in self._inverse_short_names.items():
            value = sample.get(short_name, np.nan)
            if value is None:
                row[feature_name] = np.nan
            elif self._feature_types[feature_name] == "numeric-real":
                row[feature_name] = float(value)
            else:
                row[feature_name] = int(round(float(value)))
        return pd.Series(row, index=self._feature_names)


@register(LOCAL_DISTANCE_EVALUATION_NAME)
class MaceNormalizedDistanceEvaluation(EvaluationObject):
    def __init__(self, metrics: list[str] | None = None, **kwargs):
        del kwargs
        raw_metrics = metrics or ["l0", "l1", "linf"]
        metric_mapping = {
            "l0": "distance_l0",
            "l1": "distance_l1",
            "linf": "distance_linf",
        }
        resolved_metrics = []
        for metric in raw_metrics:
            metric_name = str(metric).lower()
            if metric_name not in metric_mapping:
                raise ValueError(f"Unsupported MACE normalized metric: {metric}")
            resolved_metrics.append(metric_mapping[metric_name])
        self._metrics = resolved_metrics

    def evaluate(self, factuals, counterfactuals) -> pd.DataFrame:
        factual_features, counterfactual_features, evaluation_mask, success_mask = (
            resolve_evaluation_inputs(factuals, counterfactuals)
        )
        selected_mask = evaluation_mask & success_mask
        results: dict[str, object] = {
            metric_name: float("nan") for metric_name in self._metrics
        }
        if int(selected_mask.sum()) == 0:
            for metric_name in self._metrics:
                results[f"{metric_name}_pointwise"] = []
            return pd.DataFrame([results])

        wrapper = _LocalMaceDatasetWrapper.from_dataset(factuals)
        factual_success = factual_features.loc[selected_mask.to_numpy()]
        counterfactual_success = counterfactual_features.loc[selected_mask.to_numpy()]

        distances_by_metric = {metric_name: [] for metric_name in self._metrics}
        for row_index in factual_success.index:
            factual_sample = wrapper.feature_row_to_short_dict(
                factual_success.loc[row_index]
            )
            counterfactual_sample = wrapper.feature_row_to_short_dict(
                counterfactual_success.loc[row_index]
            )
            for metric_name in self._metrics:
                distances_by_metric[metric_name].append(
                    float(
                        normalizedDistance.getDistanceBetweenSamples(
                            factual_sample,
                            counterfactual_sample,
                            MACE_DISTANCE_NORMS[metric_name],
                            wrapper,
                        )
                    )
                )

        for metric_name, values in distances_by_metric.items():
            results[metric_name] = float(sum(values) / len(values))
            results[f"{metric_name}_pointwise"] = list(values)
        return pd.DataFrame([results])


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _apply_overrides(
    config: dict,
    norm_type: str,
    epsilon: float,
    desired_class: int | str | None,
) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["device"] = "cpu"
    cfg["method"]["device"] = "cpu"
    cfg["method"]["norm_type"] = str(norm_type)
    cfg["method"]["epsilon"] = float(epsilon)
    cfg["method"]["desired_class"] = desired_class
    cfg["name"] = (
        f"{cfg['dataset']['name']}_randomforest_mace_"
        f"{str(norm_type).lower()}_{epsilon:.0e}".replace("+", "")
    )
    cfg["logger"]["path"] = f"./logs/{cfg['name']}.log"
    evaluation_names = {item.get("name") for item in cfg.get("evaluation", [])}
    if LOCAL_DISTANCE_EVALUATION_NAME not in evaluation_names:
        cfg.setdefault("evaluation", []).append(
            {
                "name": LOCAL_DISTANCE_EVALUATION_NAME,
                "metrics": ["l0", "l1", "linf"],
            }
        )
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


def _subset_dataset(dataset, indices: pd.Index, flag_name: str):
    feature_df = dataset.get(target=False)
    target_df = dataset.get(target=True)
    subset_df = pd.concat(
        [feature_df.loc[indices], target_df.loc[indices]],
        axis=1,
    ).reindex(columns=dataset.ordered_features())
    subset = dataset.clone()
    subset.update(flag_name, True, df=subset_df)
    subset.freeze()
    return subset


def _select_factuals(dataset, target_model, desired_class, num_factuals: int):
    predictions = (
        target_model.predict(dataset, batch_size=max(1, len(dataset)))
        .argmax(dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    target_index = _resolve_target_index(target_model, desired_class)

    keep_mask = predictions != target_index
    keep_index = dataset.get(target=False).index[keep_mask]
    if keep_index.shape[0] < num_factuals:
        raise ValueError(
            f"Requested {num_factuals} factuals but only found {keep_index.shape[0]}"
        )
    selected_index = keep_index[:num_factuals]
    return _subset_dataset(dataset, selected_index, "testset")


def _evaluate(experiment: Experiment, factuals, counterfactuals) -> pd.DataFrame:
    evaluation_results = [
        evaluation_step.evaluate(factuals, counterfactuals)
        for evaluation_step in experiment._evaluation
    ]
    return pd.concat(evaluation_results, axis=1)


def _resolve_target_index(target_model, desired_class) -> int:
    class_to_index = target_model.get_class_to_index()
    if desired_class is None:
        if len(class_to_index) != 2:
            raise ValueError(
                "desired_class=None selection requires binary classification"
            )
        return 1
    return int(class_to_index[desired_class])


def _to_builtin_scalar(value):
    if pd.isna(value):
        return float("nan")
    if hasattr(value, "item"):
        return value.item()
    return value


def _build_counterfactual_dataset(factuals, counterfactual_features, desired_class):
    target_column = factuals.target_column
    counterfactual_target = pd.DataFrame(
        -1.0,
        index=counterfactual_features.index,
        columns=[target_column],
    )
    counterfactual_df = pd.concat(
        [counterfactual_features, counterfactual_target], axis=1
    ).reindex(columns=factuals.ordered_features())

    output = factuals.clone()
    output.update("counterfactual", True, df=counterfactual_df)
    if desired_class is not None:
        evaluation_filter = pd.DataFrame(
            True,
            index=counterfactual_df.index,
            columns=["evaluation_filter"],
            dtype=bool,
        )
        output.update("evaluation_filter", evaluation_filter)
    output.freeze()
    return output


def _predict_label_indices(target_model, dataset_or_features) -> np.ndarray:
    if isinstance(dataset_or_features, DatasetObject):
        prediction = target_model.predict(
            dataset_or_features,
            batch_size=max(1, len(dataset_or_features)),
        )
    else:
        prediction = target_model.get_prediction(dataset_or_features, proba=False)
    return prediction.argmax(dim=1).detach().cpu().numpy()


def _select_observable_pool(dataset, target_model, desired_class):
    predictions = _predict_label_indices(target_model, dataset)
    target_index = _resolve_target_index(target_model, desired_class)
    pool_index = dataset.get(target=False).index[predictions == target_index]
    observable_pool = dataset.get(target=False).loc[pool_index].copy(deep=True)
    if observable_pool.shape[0] == 0:
        raise ValueError("Observable pool is empty for MO baseline")
    return observable_pool


def _violates_actionability(wrapper, factual_sample, observable_sample) -> bool:
    for attr_name_kurz in wrapper.getInputAttributeNames("kurz"):
        attr_obj = wrapper.attributes_kurz[attr_name_kurz]
        factual_value = factual_sample[attr_name_kurz]
        observable_value = observable_sample[attr_name_kurz]
        if attr_obj.actionability == "none" and factual_value != observable_value:
            return True
        if (
            attr_obj.actionability == "same-or-increase"
            and factual_value > observable_value
        ):
            return True
        if (
            attr_obj.actionability == "same-or-decrease"
            and factual_value < observable_value
        ):
            return True
    return False


def _compute_mo_counterfactuals(
    experiment: Experiment, factuals, observable_pool: pd.DataFrame
):
    wrapper = _LocalMaceDatasetWrapper.from_dataset(factuals)
    factual_features = factuals.get(target=False)
    factual_labels = _predict_label_indices(experiment._target_model, factual_features)
    observable_labels = _predict_label_indices(experiment._target_model, observable_pool)

    observable_samples = []
    for row_position, row_index in enumerate(observable_pool.index):
        observable_samples.append(
            wrapper.feature_row_to_short_dict(
                observable_pool.loc[row_index],
                int(observable_labels[row_position]),
            )
        )

    rows = []
    for row_position, row_index in enumerate(factual_features.index):
        factual_sample = wrapper.feature_row_to_short_dict(
            factual_features.loc[row_index],
            int(factual_labels[row_position]),
        )
        best_sample = None
        best_distance = float("inf")
        norm_type = str(experiment._method._norm_type[0])

        for observable_sample in observable_samples:
            if observable_sample["y"] == factual_sample["y"]:
                continue
            if _violates_actionability(wrapper, factual_sample, observable_sample):
                continue

            candidate_distance = float(
                normalizedDistance.getDistanceBetweenSamples(
                    factual_sample,
                    observable_sample,
                    norm_type,
                    wrapper,
                )
            )
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_sample = observable_sample

        if best_sample is None:
            rows.append(
                pd.Series(np.nan, index=wrapper._feature_names, dtype="float64")
            )
        else:
            rows.append(wrapper.short_dict_to_feature_row(best_sample))

    counterfactual_features = pd.DataFrame(
        rows,
        index=factual_features.index,
        columns=factual_features.columns,
    )
    return _build_counterfactual_dataset(
        factuals,
        counterfactual_features,
        getattr(experiment._method, "_desired_class", None),
    )


def _compute_mace_vs_mo_comparison(
    norm_type: str,
    epsilon: float,
    mace_distances: dict[str, float],
    mo_distances: dict[str, float],
    mace_pointwise: dict[str, list[float]],
    mo_pointwise: dict[str, list[float]],
) -> dict[str, float | str]:
    optimized_metric = NORM_TO_METRIC[str(norm_type)]
    mace_distance = float(mace_distances[optimized_metric])
    mo_distance = float(mo_distances[optimized_metric])

    improvement_terms = []
    for mace_value, mo_value in zip(
        mace_pointwise[optimized_metric], mo_pointwise[optimized_metric]
    ):
        if mo_value <= 0:
            continue
        improvement_terms.append(1.0 - float(mace_value) / float(mo_value))

    if improvement_terms:
        improvement = 100.0 * float(sum(improvement_terms) / len(improvement_terms))
    else:
        improvement = float("nan")

    return {
        "optimized_metric": optimized_metric,
        "mace_distance": mace_distance,
        "mo_distance": mo_distance,
        "mace_vs_mo_improvement": improvement,
    }


def _run_single(
    config: dict,
    num_factuals: int,
) -> dict:
    baseline_experiment = Experiment(config)
    trainset, testset = _materialize_datasets(baseline_experiment)

    baseline_experiment._target_model.fit(trainset)
    factuals = _select_factuals(
        testset,
        baseline_experiment._target_model,
        getattr(baseline_experiment._method, "_desired_class", None),
        num_factuals,
    )

    observable_pool = _select_observable_pool(
        testset,
        baseline_experiment._target_model,
        getattr(baseline_experiment._method, "_desired_class", None),
    )
    mo_start_time = time.perf_counter()
    mo_counterfactuals = _compute_mo_counterfactuals(
        baseline_experiment, factuals, observable_pool
    )
    mo_generation_seconds = time.perf_counter() - mo_start_time
    mo_metrics_df = _evaluate(baseline_experiment, factuals, mo_counterfactuals)
    mo_metrics_raw = {
        key: _to_builtin_scalar(value)
        for key, value in mo_metrics_df.iloc[0].to_dict().items()
    }
    mo_metric_row = {
        key: value
        for key, value in mo_metrics_raw.items()
        if not str(key).endswith("_pointwise")
    }
    mo_pointwise = {
        key.replace("_pointwise", ""): list(value)
        for key, value in mo_metrics_raw.items()
        if str(key).endswith("_pointwise")
    }
    mo_distances = {
        key: float(mo_metric_row[key])
        for key in MACE_DISTANCE_NORMS
    }

    experiment_cfg = deepcopy(config)
    preprocess_cfg = [
        step
        for step in experiment_cfg.get("preprocess", [])
        if step.get("name", "").lower() != "finalize"
    ]
    preprocess_cfg.append(
        {
            "name": LOCAL_SELECT_PREPROCESS_NAME,
            "indices": factuals.get(target=False).index.tolist(),
        }
    )
    preprocess_cfg.append({"name": "finalize"})
    experiment_cfg["preprocess"] = preprocess_cfg
    experiment = Experiment(experiment_cfg)
    start_time = time.perf_counter()
    metrics = experiment.run()
    generation_seconds = time.perf_counter() - start_time
    metric_row_raw = {
        key: _to_builtin_scalar(value)
        for key, value in metrics.iloc[0].to_dict().items()
    }
    metric_row = {
        key: value
        for key, value in metric_row_raw.items()
        if not str(key).endswith("_pointwise")
    }
    mace_pointwise = {
        key.replace("_pointwise", ""): list(value)
        for key, value in metric_row_raw.items()
        if str(key).endswith("_pointwise")
    }
    mace_distances = {
        key: float(metric_row[key])
        for key in MACE_DISTANCE_NORMS
    }
    comparison = _compute_mace_vs_mo_comparison(
        norm_type=str(config["method"]["norm_type"]),
        epsilon=float(config["method"]["epsilon"]),
        mace_distances=mace_distances,
        mo_distances=mo_distances,
        mace_pointwise=mace_pointwise,
        mo_pointwise=mo_pointwise,
    )

    return {
        "config_name": config["name"],
        "dataset": config["dataset"]["name"],
        "norm_type": config["method"]["norm_type"],
        "epsilon": float(config["method"]["epsilon"]),
        "num_factuals": int(len(factuals)),
        "generation_seconds": float(generation_seconds),
        "metrics": metric_row,
        "mo_generation_seconds": float(mo_generation_seconds),
        "mo_metrics": mo_metric_row,
        "comparison": comparison,
    }


TABLE3_FOREST_MO_REFERENCES = {
    1.0e-5: {
        "adult": {"zero_norm": 51.0, "one_norm": 82.0, "infty_norm": 71.0},
        "credit": {"zero_norm": 68.0, "one_norm": 97.0, "infty_norm": 96.0},
        "compas": {"zero_norm": 1.0, "one_norm": 6.0, "infty_norm": 6.0},
    },
    1.0e-3: {
        "adult": {"zero_norm": 51.0, "one_norm": 81.0, "infty_norm": 69.0},
        "credit": {"zero_norm": 68.0, "one_norm": 61.0, "infty_norm": 38.0},
        "compas": {"zero_norm": 1.0, "one_norm": 6.0, "infty_norm": 6.0},
    },
}


def _paper_reference_improvement(
    dataset: str,
    norm_type: str,
    epsilon: float,
) -> float | None:
    epsilon_key = float(epsilon)
    if epsilon_key not in TABLE3_FOREST_MO_REFERENCES:
        return None
    dataset_refs = TABLE3_FOREST_MO_REFERENCES[epsilon_key]
    if dataset not in dataset_refs:
        return None
    return dataset_refs[dataset].get(norm_type)


def _assert_result(result: dict, expected_num_factuals: int, epsilon: float) -> None:
    if int(result["num_factuals"]) != int(expected_num_factuals):
        raise AssertionError(
            "Selected factual count does not match requested count: "
            f"{result['num_factuals']} vs {expected_num_factuals}"
        )

    metrics = result["metrics"]
    mo_metrics = result["mo_metrics"]
    comparison = result["comparison"]

    if float(metrics["validity"]) != 1.0:
        raise AssertionError(f"Expected MACE validity == 1.0, got {metrics['validity']}")
    if float(mo_metrics["validity"]) != 1.0:
        raise AssertionError(f"Expected MO validity == 1.0, got {mo_metrics['validity']}")

    for name, value in metrics.items():
        if name.startswith("distance_") and pd.isna(value):
            raise AssertionError(f"MACE metric {name} is NaN")
    for name, value in mo_metrics.items():
        if name.startswith("distance_") and pd.isna(value):
            raise AssertionError(f"MO metric {name} is NaN")

    if float(comparison["mace_distance"]) > float(comparison["mo_distance"]) + float(
        epsilon
    ) + 1e-8:
        raise AssertionError(
            f"{comparison['optimized_metric']} expected MACE <= MO + epsilon, got "
            f"{comparison['mace_distance']} vs {comparison['mo_distance']}"
        )

    improvement = comparison["mace_vs_mo_improvement"]
    if not pd.isna(improvement) and float(improvement) < -1e-8:
        raise AssertionError(
            "MACE vs MO improvement must be >= 0, "
            f"got {comparison['mace_vs_mo_improvement']}"
        )


def _iter_jobs(
    datasets: list[str],
    norms: list[str],
    epsilon: float,
    desired_class: int | str | None,
    num_factuals: int,
):
    for dataset in datasets:
        base_config = _load_config(DEFAULT_CONFIG_PATHS[dataset])
        for norm_type in norms:
            yield {
                "dataset": dataset,
                "norm_type": norm_type,
                "epsilon": float(epsilon),
                "desired_class": desired_class,
                "num_factuals": int(num_factuals),
                "config": _apply_overrides(
                    base_config,
                    norm_type=norm_type,
                    epsilon=epsilon,
                    desired_class=desired_class,
                ),
            }


def _build_summary(results: list[dict], errors: list[dict]) -> dict:
    summary_table = []
    successful_results = []
    for result in results:
        assert_passed = bool(result.get("assert_passed", False))
        paper_ref = _paper_reference_improvement(
            dataset=str(result["dataset"]),
            norm_type=str(result["norm_type"]),
            epsilon=float(result["epsilon"]),
        )
        summary_table.append(
            {
                "dataset": result["dataset"],
                "norm_type": result["norm_type"],
                "epsilon": result["epsilon"],
                "num_factuals": result["num_factuals"],
                "mace_validity": result["metrics"]["validity"],
                "mo_validity": result["mo_metrics"]["validity"],
                "optimized_metric": result["comparison"]["optimized_metric"],
                "mace_distance": result["comparison"]["mace_distance"],
                "mo_distance": result["comparison"]["mo_distance"],
                "mace_vs_mo_improvement": result["comparison"][
                    "mace_vs_mo_improvement"
                ],
                "paper_improvement_reference": paper_ref,
                "assert_passed": assert_passed,
                "assert_error": result.get("assert_error"),
            }
        )
        if assert_passed:
            successful_results.append(result)

    aggregate: dict[str, object] = {
        "num_jobs": len(results) + len(errors),
        "num_results": len(results),
        "num_errors": len(errors),
        "num_assert_passed": sum(1 for result in results if result.get("assert_passed")),
    }

    if successful_results:
        aggregate["overall"] = {
            "mean_mace_distance": float(
                sum(
                    float(result["comparison"]["mace_distance"])
                    for result in successful_results
                )
                / len(successful_results)
            ),
            "mean_mo_distance": float(
                sum(
                    float(result["comparison"]["mo_distance"])
                    for result in successful_results
                )
                / len(successful_results)
            ),
            "mean_mace_vs_mo_improvement": float(
                sum(
                    float(result["comparison"]["mace_vs_mo_improvement"])
                    for result in successful_results
                    if not pd.isna(result["comparison"]["mace_vs_mo_improvement"])
                )
                / max(
                    1,
                    sum(
                        1
                        for result in successful_results
                        if not pd.isna(result["comparison"]["mace_vs_mo_improvement"])
                    ),
                )
            ),
        }

        per_dataset = {}
        for dataset in sorted({result["dataset"] for result in successful_results}):
            dataset_results = [
                result
                for result in successful_results
                if result["dataset"] == dataset
            ]
            per_dataset[dataset] = {
                "mean_mace_distance": float(
                    sum(float(item["comparison"]["mace_distance"]) for item in dataset_results)
                    / len(dataset_results)
                ),
                "mean_mo_distance": float(
                    sum(float(item["comparison"]["mo_distance"]) for item in dataset_results)
                    / len(dataset_results)
                ),
            }
        aggregate["per_dataset"] = per_dataset

    return {
        "summary_table": summary_table,
        "aggregate": aggregate,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DEFAULT_CONFIG_PATHS), action="append")
    parser.add_argument("--datasets", choices=sorted(DEFAULT_CONFIG_PATHS), nargs="+")
    parser.add_argument("--norm", choices=sorted(NORM_TO_METRIC), action="append")
    parser.add_argument("--norms", choices=sorted(NORM_TO_METRIC), nargs="+")
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--desired-class", type=int, default=1)
    parser.add_argument("--num-factuals", type=int, default=500)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--continue-on-error", dest="continue_on_error", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.set_defaults(continue_on_error=True)
    args = parser.parse_args()

    datasets = args.datasets or args.dataset or list(DEFAULT_CONFIG_PATHS)
    norms = args.norms or args.norm or list(NORM_TO_METRIC)

    results: list[dict] = []
    errors: list[dict] = []

    for job in _iter_jobs(
        datasets=datasets,
        norms=norms,
        epsilon=args.epsilon,
        desired_class=args.desired_class,
        num_factuals=args.num_factuals,
    ):
        try:
            result = _run_single(job["config"], job["num_factuals"])
            try:
                _assert_result(result, job["num_factuals"], job["epsilon"])
                result["assert_passed"] = True
                result["assert_error"] = None
            except Exception as error:
                result["assert_passed"] = False
                result["assert_error"] = str(error)
                if args.strict:
                    raise
            results.append(result)
        except Exception as error:
            error_record = {
                "dataset": job["dataset"],
                "norm_type": job["norm_type"],
                "epsilon": job["epsilon"],
                "num_factuals": job["num_factuals"],
                "error_message": str(error),
            }
            errors.append(error_record)
            if args.strict:
                payload = {
                    "results": results,
                    **_build_summary(results, errors),
                }
                rendered = json.dumps(payload, indent=2, sort_keys=True)
                if args.output_json:
                    Path(args.output_json).write_text(rendered, encoding="utf-8")
                print(rendered)
                raise SystemExit(1)
            if not args.continue_on_error:
                payload = {
                    "results": results,
                    **_build_summary(results, errors),
                }
                rendered = json.dumps(payload, indent=2, sort_keys=True)
                if args.output_json:
                    Path(args.output_json).write_text(rendered, encoding="utf-8")
                print(rendered)
                raise SystemExit(1)

    payload = {
        "results": results,
        **_build_summary(results, errors),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(rendered, encoding="utf-8")
    print(rendered)

    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
