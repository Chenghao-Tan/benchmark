from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset.german.german import GermanDataset
from experiment import Experiment
from method.rbr.rbr import RbrMethod
from model.mlp.mlp import MlpModel
from preprocess.preprocess_object import PreProcessObject
from utils.registry import get_registry, register

DEFAULT_CURRENT_CONFIG = "./experiment/rbr/german_mlp_rbr_reproduce_current.yaml"
DEFAULT_FUTURE_CONFIG = "./experiment/rbr/german_mlp_rbr_reproduce_future.yaml"
LOCAL_SELECT_PREPROCESS_NAME = "rbr_select_columns_local"
LOCAL_MERGE_PREPROCESS_NAME = "rbr_merge_dataset_local"
PAPER_GERMAN_METRICS = {
    "present_accuracy": "0.67 ± 0.02",
    "present_auc": "0.60 ± 0.03",
    "shift_accuracy": "0.66 ± 0.23",
    "shift_auc": "0.60 ± 0.04",
}
RECOURSE_BENCHMARKS_CURRENT_VALIDITY_MIN = 0.9
RECOURSE_BENCHMARKS_FUTURE_VALIDITY_MIN = 0.7
NUMERICAL_FEATURES = ["age", "amount", "duration"]
TARGET_COLUMN = "credit_risk"
CATEGORICAL_FEATURE = "personal_status_sex"
SELECTED_COLUMNS = ["duration", "amount", "age", CATEGORICAL_FEATURE]


def _snapshot_dataset(dataset) -> pd.DataFrame:
    if getattr(dataset, "_freeze", False):
        return pd.concat([dataset.get(target=False), dataset.get(target=True)], axis=1)
    return dataset.snapshot()


@register(LOCAL_SELECT_PREPROCESS_NAME)
class SelectColumnsPreProcess(PreProcessObject):
    def __init__(self, seed: int | None = None, columns: list[str] | None = None, **kwargs):
        del kwargs
        self._seed = seed
        self._columns = list(columns or SELECTED_COLUMNS)

    def transform(self, input):
        df = _snapshot_dataset(input)
        selected_columns = [*self._columns, input.target_column]
        missing = [column for column in selected_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for SelectColumnsPreProcess: {missing}")
        output = input if not getattr(input, "_freeze", False) else input.clone()
        output.update(
            "selected_columns",
            list(self._columns),
            df=df.loc[:, selected_columns].copy(deep=True),
        )
        return output


@register(LOCAL_MERGE_PREPROCESS_NAME)
class MergeDatasetPreProcess(PreProcessObject):
    def __init__(
        self,
        seed: int | None = None,
        merge_dataset_name: str = "german_roar",
        merge_dataset_path: str | None = None,
        columns: list[str] | None = None,
        train_fraction: float | None = None,
        random_state: int | None = None,
        dataset_flag: str | None = None,
        **kwargs,
    ):
        del kwargs
        self._seed = seed
        self._merge_dataset_name = str(merge_dataset_name)
        self._merge_dataset_path = merge_dataset_path
        self._columns = list(columns or SELECTED_COLUMNS)
        self._train_fraction = (
            None if train_fraction is None else float(train_fraction)
        )
        self._random_state = random_state
        self._dataset_flag = dataset_flag

    def transform(self, input):
        merge_cfg = {"path": self._merge_dataset_path} if self._merge_dataset_path else {}
        dataset_cls = get_registry("dataset")[self._merge_dataset_name]
        merge_dataset = dataset_cls(**merge_cfg)
        merge_df = merge_dataset.snapshot().loc[
            :, [*self._columns, merge_dataset.target_column]
        ].copy(deep=True)
        if self._train_fraction is not None:
            if not 0.0 < self._train_fraction < 1.0:
                raise ValueError("train_fraction must satisfy 0 < value < 1")
            X = merge_df.drop(columns=[merge_dataset.target_column])
            y = merge_df[merge_dataset.target_column].astype(int)
            merge_X, _, merge_y, _ = train_test_split(
                X,
                y,
                train_size=self._train_fraction,
                random_state=self._random_state,
                stratify=y,
            )
            merge_df = pd.concat(
                [merge_X, merge_y.rename(merge_dataset.target_column)],
                axis=1,
            ).loc[:, [*self._columns, merge_dataset.target_column]]

        input_df = _snapshot_dataset(input).loc[
            :, [*self._columns, input.target_column]
        ].copy(deep=True)
        merged_df = pd.concat([input_df, merge_df], ignore_index=True)

        output = input.clone()
        output.update("merged_dataset", True, df=merged_df)
        if self._dataset_flag is not None:
            output.update(self._dataset_flag, True)
        return output


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_reference_transformer(
    reference_df: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], dict[str, list[str]]]:
    combined = reference_df.drop(columns=[TARGET_COLUMN]).copy(deep=True)
    categorical_features = [
        column
        for column in combined.columns
        if column not in NUMERICAL_FEATURES
    ]

    try:
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", categorical_encoder, categorical_features),
        ],
        sparse_threshold=0.0,
    )
    transformer.fit(combined)

    cat_transformer = transformer.named_transformers_["cat"]
    categories = list(cat_transformer.categories_[0].tolist())
    cat_columns = [f"{CATEGORICAL_FEATURE}_cat_{category}" for category in categories]
    feature_names = list(NUMERICAL_FEATURES) + cat_columns
    encoding_map = {CATEGORICAL_FEATURE: cat_columns}
    return transformer, feature_names, encoding_map


def _materialize_datasets(experiment: Experiment):
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


def _materialize_single_dataset(experiment: Experiment):
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
    if len(datasets) != 1:
        raise ValueError("Expected exactly one dataset after preprocessing")
    return datasets[0]


def _transform_features(
    transformer: ColumnTransformer,
    X: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    transformed = transformer.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return pd.DataFrame(
        transformed,
        index=X.index,
        columns=feature_names,
    )


def _build_processed_metadata(
    template_dataset: GermanDataset | GermanRoarDataset,
    feature_names: list[str],
    encoding_map: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
    raw_feature_type = template_dataset.attr("raw_feature_type")
    raw_feature_mutability = template_dataset.attr("raw_feature_mutability")
    raw_feature_actionability = template_dataset.attr("raw_feature_actionability")

    encoded_feature_type: dict[str, str] = {}
    encoded_feature_mutability: dict[str, bool] = {}
    encoded_feature_actionability: dict[str, str] = {}

    categorical_columns = set(encoding_map[CATEGORICAL_FEATURE])
    for feature_name in feature_names:
        if feature_name in categorical_columns:
            encoded_feature_type[feature_name] = "binary"
            encoded_feature_mutability[feature_name] = bool(
                raw_feature_mutability[CATEGORICAL_FEATURE]
            )
            encoded_feature_actionability[feature_name] = str(
                raw_feature_actionability[CATEGORICAL_FEATURE]
            )
        else:
            encoded_feature_type[feature_name] = str(raw_feature_type[feature_name])
            encoded_feature_mutability[feature_name] = bool(
                raw_feature_mutability[feature_name]
            )
            encoded_feature_actionability[feature_name] = str(
                raw_feature_actionability[feature_name]
            )
    return (
        encoded_feature_type,
        encoded_feature_mutability,
        encoded_feature_actionability,
    )


def _make_frozen_processed_dataset(
    template_dataset: GermanDataset | GermanRoarDataset,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    encoding_map: dict[str, list[str]],
    dataset_flag: str,
) -> object:
    dataset = template_dataset.clone()
    combined = pd.concat(
        [X.loc[:, feature_names], y.rename(TARGET_COLUMN)],
        axis=1,
    )
    combined = combined.loc[:, [*feature_names, TARGET_COLUMN]]

    (
        encoded_feature_type,
        encoded_feature_mutability,
        encoded_feature_actionability,
    ) = _build_processed_metadata(
        template_dataset,
        feature_names,
        encoding_map,
    )

    dataset.update("encoding", deepcopy(encoding_map), df=combined)
    dataset.update("encoded_feature_type", encoded_feature_type)
    dataset.update("encoded_feature_mutability", encoded_feature_mutability)
    dataset.update("encoded_feature_actionability", encoded_feature_actionability)
    dataset.update(dataset_flag, True)
    dataset.freeze()
    return dataset


def _build_experiment_config(
    config: dict,
    preprocess_steps: list[dict],
) -> dict:
    experiment_cfg = deepcopy(config)
    experiment_cfg["preprocess"] = deepcopy(preprocess_steps)
    experiment_cfg.setdefault("evaluation", [])
    return experiment_cfg


def _compute_model_metrics(model: MlpModel, testset) -> dict[str, float]:
    probabilities = model.predict_proba(testset).detach().cpu().numpy()
    prediction = probabilities.argmax(axis=1)

    y = testset.get(target=True).iloc[:, 0]
    class_to_index = model.get_class_to_index()
    encoded_target = np.array(
        [class_to_index[int(value)] for value in y.astype(int).tolist()],
        dtype=np.int64,
    )

    accuracy = float(np.mean(prediction == encoded_target))
    positive_index = class_to_index.get(1, max(class_to_index.values()))
    if len(set(encoded_target.tolist())) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(encoded_target, probabilities[:, positive_index]))
    return {"test_accuracy": accuracy, "test_auc": auc}


def _select_recourse_factuals(model: MlpModel, testset, experiment_cfg: dict):
    factual_selection = str(
        experiment_cfg.get("factual_selection", "all")
    ).lower()
    prediction_probabilities = model.predict_proba(testset).detach().cpu().numpy()
    prediction_indices = pd.Series(
        prediction_probabilities.argmax(axis=1),
        index=testset.get(target=False).index,
        dtype="int64",
    )
    desired_index = model.get_class_to_index()[1]
    negative_mask = prediction_indices.ne(desired_index)

    if factual_selection == "all":
        keep_mask = negative_mask
    elif factual_selection == "negative_class":
        num_factuals = int(experiment_cfg.get("num_factuals", 5))
        negative_indices = prediction_indices.index[negative_mask.to_numpy()]
        if len(negative_indices) < num_factuals:
            raise ValueError("Not enough negative-class factuals for sampling")
        chosen = (
            pd.Series(negative_indices)
            .sample(n=num_factuals, random_state=42)
            .tolist()
        )
        keep_mask = pd.Series(False, index=prediction_indices.index, dtype=bool)
        keep_mask.loc[chosen] = True
    else:
        raise ValueError(f"Unsupported factual_selection: {factual_selection}")

    factuals = testset.clone()
    factual_df = pd.concat([testset.get(target=False), testset.get(target=True)], axis=1)
    factual_df = factual_df.loc[keep_mask].copy(deep=True)
    factuals.update("reproduce_factuals", True, df=factual_df)
    factuals.freeze()
    if len(factuals) == 0:
        raise ValueError("No factuals selected for RBR reproduction")
    return factuals


def _compute_distance_metrics(factuals, counterfactuals) -> tuple[dict[str, float], int]:
    factual_features = factuals.get(target=False)
    counterfactual_features = counterfactuals.get(target=False).reindex(
        index=factual_features.index,
        columns=factual_features.columns,
    )
    success_mask = ~counterfactual_features.isna().any(axis=1)
    successful_count = int(success_mask.sum())
    if successful_count == 0:
        return (
            {
                "distance_l0": float("nan"),
                "distance_l1": float("nan"),
                "distance_l2": float("nan"),
                "distance_linf": float("nan"),
                "l1_cost": float("nan"),
            },
            0,
        )

    delta = (
        counterfactual_features.loc[success_mask].to_numpy(dtype=np.float64)
        - factual_features.loc[success_mask].to_numpy(dtype=np.float64)
    )
    l0 = np.sum(~np.isclose(delta, np.zeros_like(delta), atol=1e-05), axis=1)
    l1 = np.sum(np.abs(delta), axis=1, dtype=np.float32)
    l2 = np.linalg.norm(delta, ord=2, axis=1)
    linf = np.max(np.abs(delta), axis=1).astype(np.float32)
    return (
        {
            "distance_l0": float(np.mean(l0)),
            "distance_l1": float(np.mean(l1)),
            "distance_l2": float(np.mean(l2)),
            "distance_linf": float(np.mean(linf)),
            "l1_cost": float(np.mean(l1)),
        },
        successful_count,
    )


def _compute_current_validity(current_model: MlpModel, counterfactuals) -> float:
    counterfactual_features = counterfactuals.get(target=False)
    success_mask = ~counterfactual_features.isna().any(axis=1)
    denominator = int(len(counterfactual_features))
    if denominator == 0:
        return float("nan")
    if int(success_mask.sum()) == 0:
        return 0.0

    probabilities = current_model.get_prediction(
        counterfactual_features.loc[success_mask],
        proba=True,
    ).detach().cpu().numpy()
    positive_probability = probabilities[:, current_model.get_class_to_index()[1]]
    return float(np.sum(positive_probability >= 0.5) / denominator)


def _compute_future_validity(
    future_models: list[MlpModel],
    counterfactuals,
) -> float:
    counterfactual_features = counterfactuals.get(target=False)
    success_mask = ~counterfactual_features.isna().any(axis=1)
    denominator = int(len(counterfactual_features))
    if denominator == 0:
        return float("nan")
    if int(success_mask.sum()) == 0:
        return 0.0

    cf_success = counterfactual_features.loc[success_mask]
    validities = []
    for future_model in future_models:
        probabilities = future_model.get_prediction(
            cf_success,
            proba=True,
        ).detach().cpu().numpy()
        positive_probability = probabilities[:, future_model.get_class_to_index()[1]]
        validities.append((positive_probability >= 0.5).astype(np.float32))
    stacked = np.stack(validities, axis=0)
    per_instance_future_validity = stacked.mean(axis=0)
    return float(np.sum(per_instance_future_validity) / denominator)


def _build_mlp_from_config(config: dict, device: str) -> MlpModel:
    model_cfg = deepcopy(config["model"])
    return MlpModel(
        seed=model_cfg.get("seed", 42),
        device=device,
        epochs=model_cfg.get("epochs", 1000),
        learning_rate=model_cfg.get("learning_rate", 0.001),
        batch_size=model_cfg.get("batch_size"),
        layers=model_cfg.get("layers"),
        optimizer=model_cfg.get("optimizer", "adam"),
        criterion=model_cfg.get("criterion", "bce"),
        output_activation=model_cfg.get("output_activation", "sigmoid"),
        pretrained_path=model_cfg.get("pretrained_path"),
        save_name=model_cfg.get("save_name"),
        weight_decay=model_cfg.get("weight_decay", 0.0),
        loss_reduction=model_cfg.get("loss_reduction", "mean"),
        xavier_uniform_init=model_cfg.get("xavier_uniform_init", False),
        early_stop_tol=model_cfg.get("early_stop_tol"),
        early_stop_patience=model_cfg.get("early_stop_patience"),
    )


def _build_rbr_from_config(
    config: dict,
    target_model: MlpModel,
    device: str,
) -> RbrMethod:
    method_cfg = deepcopy(config["method"])
    return RbrMethod(
        target_model=target_model,
        seed=method_cfg.get("seed", 42),
        device=device,
        desired_class=method_cfg.get("desired_class", 1),
        num_samples=method_cfg.get("num_samples", 200),
        perturb_radius=method_cfg.get("perturb_radius", 0.2),
        delta_plus=method_cfg.get("delta_plus", 1.0),
        sigma=method_cfg.get("sigma", 1.0),
        epsilon_op=method_cfg.get("epsilon_op", 0.5),
        epsilon_pe=method_cfg.get("epsilon_pe", 1.0),
        max_iter=method_cfg.get("max_iter", 1000),
        clamp=method_cfg.get("clamp", False),
        enforce_encoding=method_cfg.get("enforce_encoding", False),
        random_state=method_cfg.get("random_state", 42),
        verbose=method_cfg.get("verbose", False),
    )


def _build_future_trainsets(
    template_dataset: GermanDataset,
    transformer: ColumnTransformer,
    feature_names: list[str],
    encoding_map: dict[str, list[str]],
    current_trainset_raw,
    future_cfg: dict,
) -> list[object]:
    experiment_cfg = future_cfg.get("experiment", {})
    shifted_train_fraction = float(experiment_cfg.get("shifted_train_fraction", 0.5))
    shifted_model_random_states = list(
        experiment_cfg.get("shifted_model_random_states", [1, 2, 3, 4, 5])
    )

    future_trainsets = []
    for random_state in shifted_model_random_states:
        merged_raw_dataset = MergeDatasetPreProcess(
            merge_dataset_name="german_roar",
            columns=SELECTED_COLUMNS,
            train_fraction=shifted_train_fraction,
            random_state=int(random_state),
            dataset_flag="trainset",
        ).transform(current_trainset_raw)
        merged_raw_df = _snapshot_dataset(merged_raw_dataset)
        future_X_raw = merged_raw_df.drop(columns=[TARGET_COLUMN])
        future_y = merged_raw_df[TARGET_COLUMN].astype(int)
        future_X = _transform_features(transformer, future_X_raw, feature_names)
        future_trainsets.append(
            _make_frozen_processed_dataset(
                template_dataset=template_dataset,
                X=future_X,
                y=future_y,
                feature_names=feature_names,
                encoding_map=encoding_map,
                dataset_flag="trainset",
            )
        )
    return future_trainsets


def _build_summary(
    current_model_metrics: dict[str, float],
    future_model_metrics: dict[str, float],
    distance_metrics: dict[str, float],
    num_factuals: int,
    num_successful: int,
    current_validity: float,
    future_validity: float,
    device: str,
) -> dict[str, float | int | str]:
    return {
        "device": device,
        "num_factuals": num_factuals,
        "num_successful": num_successful,
        "current_validity": current_validity,
        "future_validity": future_validity,
        **distance_metrics,
        "current_model_test_accuracy": current_model_metrics["test_accuracy"],
        "current_model_test_auc": current_model_metrics["test_auc"],
        "future_model_test_accuracy": future_model_metrics["test_accuracy"],
        "future_model_test_auc": future_model_metrics["test_auc"],
    }


def _assert_summary(summary: dict[str, float | int | str]) -> None:
    current_validity = float(summary["current_validity"])
    future_validity = float(summary["future_validity"])
    assert current_validity >= RECOURSE_BENCHMARKS_CURRENT_VALIDITY_MIN, (
        "current_validity does not satisfy recourse_benchmarks RBR threshold: "
        f"{current_validity} < {RECOURSE_BENCHMARKS_CURRENT_VALIDITY_MIN}"
    )
    assert future_validity >= RECOURSE_BENCHMARKS_FUTURE_VALIDITY_MIN, (
        "future_validity does not satisfy recourse_benchmarks RBR threshold: "
        f"{future_validity} < {RECOURSE_BENCHMARKS_FUTURE_VALIDITY_MIN}"
    )


def run_reproduction(
    current_config_path: str = DEFAULT_CURRENT_CONFIG,
    future_config_path: str = DEFAULT_FUTURE_CONFIG,
) -> dict[str, float | int | str]:
    device = _resolve_device()
    current_cfg = _load_config((PROJECT_ROOT / current_config_path).resolve())
    future_cfg = _load_config((PROJECT_ROOT / future_config_path).resolve())

    current_preprocess = [
        {"name": LOCAL_SELECT_PREPROCESS_NAME, "columns": SELECTED_COLUMNS},
        {
            "name": "split",
            "seed": int(current_cfg.get("experiment", {}).get("split_random_state", 42)),
            "split": 1.0 - float(current_cfg.get("experiment", {}).get("train_split", 0.8)),
        },
        {"name": "finalize"},
    ]
    reference_preprocess = [
        {"name": LOCAL_SELECT_PREPROCESS_NAME, "columns": SELECTED_COLUMNS},
        {
            "name": LOCAL_MERGE_PREPROCESS_NAME,
            "merge_dataset_name": "german_roar",
            "columns": SELECTED_COLUMNS,
        },
        {"name": "finalize"},
    ]

    current_raw_experiment = Experiment(
        _build_experiment_config(current_cfg, current_preprocess)
    )
    current_trainset_raw, current_testset_raw = _materialize_datasets(current_raw_experiment)

    reference_experiment = Experiment(
        _build_experiment_config(current_cfg, reference_preprocess)
    )
    reference_dataset = _materialize_single_dataset(reference_experiment)
    transformer, feature_names, encoding_map = _build_reference_transformer(
        _snapshot_dataset(reference_dataset)
    )

    current_train_raw_df = _snapshot_dataset(current_trainset_raw)
    current_test_raw_df = _snapshot_dataset(current_testset_raw)
    current_X_train = _transform_features(
        transformer,
        current_train_raw_df.drop(columns=[TARGET_COLUMN]),
        feature_names,
    )
    current_X_test = _transform_features(
        transformer,
        current_test_raw_df.drop(columns=[TARGET_COLUMN]),
        feature_names,
    )

    current_template_dataset = GermanDataset()
    current_trainset = _make_frozen_processed_dataset(
        template_dataset=current_template_dataset,
        X=current_X_train,
        y=current_train_raw_df[TARGET_COLUMN].astype(int),
        feature_names=feature_names,
        encoding_map=encoding_map,
        dataset_flag="trainset",
    )
    current_testset = _make_frozen_processed_dataset(
        template_dataset=current_template_dataset,
        X=current_X_test,
        y=current_test_raw_df[TARGET_COLUMN].astype(int),
        feature_names=feature_names,
        encoding_map=encoding_map,
        dataset_flag="testset",
    )

    current_model = _build_mlp_from_config(current_cfg, device=device)
    current_model.fit(current_trainset)
    current_model_metrics = _compute_model_metrics(current_model, current_testset)

    rbr_method = _build_rbr_from_config(current_cfg, current_model, device=device)
    rbr_method.fit(current_trainset)
    factuals = _select_recourse_factuals(
        current_model,
        current_testset,
        current_cfg.get("experiment", {}),
    )
    counterfactuals = rbr_method.predict(factuals)

    future_trainsets = _build_future_trainsets(
        template_dataset=current_template_dataset,
        transformer=transformer,
        feature_names=feature_names,
        encoding_map=encoding_map,
        current_trainset_raw=current_trainset_raw,
        future_cfg=future_cfg,
    )
    future_models = []
    future_model_metrics = []
    for future_trainset in future_trainsets:
        future_model = _build_mlp_from_config(future_cfg, device=device)
        future_model.fit(future_trainset)
        future_models.append(future_model)
        future_model_metrics.append(_compute_model_metrics(future_model, current_testset))

    mean_future_model_metrics = {
        "test_accuracy": float(
            np.mean([metric["test_accuracy"] for metric in future_model_metrics])
        ),
        "test_auc": float(
            np.mean([metric["test_auc"] for metric in future_model_metrics])
        ),
    }

    distance_metrics, num_successful = _compute_distance_metrics(
        factuals,
        counterfactuals,
    )
    current_validity = _compute_current_validity(current_model, counterfactuals)
    future_validity = _compute_future_validity(future_models, counterfactuals)

    summary = _build_summary(
        current_model_metrics=current_model_metrics,
        future_model_metrics=mean_future_model_metrics,
        distance_metrics=distance_metrics,
        num_factuals=len(factuals),
        num_successful=num_successful,
        current_validity=current_validity,
        future_validity=future_validity,
        device=device,
    )
    _assert_summary(summary)
    return summary


def test_run_experiment():
    return run_reproduction()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-config", default=DEFAULT_CURRENT_CONFIG)
    parser.add_argument("--future-config", default=DEFAULT_FUTURE_CONFIG)
    args = parser.parse_args()

    summary = run_reproduction(
        current_config_path=args.current_config,
        future_config_path=args.future_config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
