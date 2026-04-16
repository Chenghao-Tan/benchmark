from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from experiment import Experiment
from preprocess.preprocess_object import PreProcessObject
from utils.registry import register

PREPARE_CONFIG_PATH = Path(__file__).with_name(
    "credit_cchvae_sklearn_logistic_regression_cchvae_prepare.yaml"
)
RUN_CONFIG_PATH = Path(__file__).with_name(
    "credit_cchvae_sklearn_logistic_regression_cchvae_reproduce.yaml"
)
ARTIFACT_DIR = PROJECT_ROOT / "cache" / "cchvae"
TRAIN_LIMIT_PATH = ARTIFACT_DIR / "train_limit.txt"
TEST_INDEX_PATH = ARTIFACT_DIR / "selected_test_indices.json"
REFERENCE_PATH = ARTIFACT_DIR / "positive_reference.csv"
LOCAL_OUTLIER_EVALUATION_NAME = "cchvae_outlier_rate_local"
LOCAL_CONNECTEDNESS_EVALUATION_NAME = "cchvae_connectedness_local"
LOCAL_LIMIT_TRAIN_PREPROCESS_NAME = "cchvae_limit_train_local"
LOCAL_SELECT_TEST_PREPROCESS_NAME = "cchvae_select_test_local"
LOCAL_ATTACH_REFERENCE_PREPROCESS_NAME = "cchvae_attach_reference_local"
DEFAULT_TRAIN_SAMPLE_LIMIT: int | None = None
DEFAULT_NCOUNTERFACTUALS: int | None = None
LOF_NEIGHBORS = [5, 10, 20, 50]
CONNECTEDNESS_EPSILONS = [10, 20, 30, 40, 50]
DBSCAN_MIN_SAMPLES = 5
OUTLIER_LIMIT = 0.2
CONNECTEDNESS_EPS = 30
CONNECTEDNESS_LIMIT = 0.5


@register(LOCAL_LIMIT_TRAIN_PREPROCESS_NAME)
class CchvaeLimitTrainPreProcess(PreProcessObject):
    def __init__(self, path: str, seed: int | None = None, **kwargs):
        del kwargs
        self._path = Path(path)
        self._seed = seed

    def transform(self, input):
        del self._seed
        if not getattr(input, "trainset", False) or not self._path.exists():
            return input
        limit = int(self._path.read_text(encoding="utf-8").strip())
        if limit >= len(input.snapshot()):
            return input
        input.update(
            "limited_trainset",
            limit,
            df=input.snapshot().iloc[:limit].copy(deep=True),
        )
        return input


@register(LOCAL_SELECT_TEST_PREPROCESS_NAME)
class CchvaeSelectTestPreProcess(PreProcessObject):
    def __init__(self, path: str, seed: int | None = None, **kwargs):
        del kwargs
        self._path = Path(path)
        self._seed = seed

    def transform(self, input):
        del self._seed
        if not getattr(input, "testset", False) or not self._path.exists():
            return input
        indices = json.loads(self._path.read_text(encoding="utf-8"))
        input.update(
            "selected_test_indices",
            indices,
            df=input.snapshot().loc[indices].copy(deep=True),
        )
        return input


@register(LOCAL_ATTACH_REFERENCE_PREPROCESS_NAME)
class CchvaeAttachReferencePreProcess(PreProcessObject):
    def __init__(self, path: str, attr_name: str = "cchvae_positive_reference", seed: int | None = None, **kwargs):
        del kwargs
        self._path = Path(path)
        self._attr_name = attr_name
        self._seed = seed

    def transform(self, input):
        del self._seed
        if not getattr(input, "testset", False) or not self._path.exists():
            return input
        input.update(self._attr_name, pd.read_csv(self._path))
        return input


@register(LOCAL_OUTLIER_EVALUATION_NAME)
class CchvaeOutlierRateEvaluation(EvaluationObject):
    def __init__(self, neighbors: list[int] | None = None, **kwargs):
        del kwargs
        self._neighbors = list(neighbors or LOF_NEIGHBORS)

    def evaluate(self, factuals, counterfactuals) -> pd.DataFrame:
        reference = factuals.attr("cchvae_positive_reference")
        _, counterfactual_features, evaluation_mask, success_mask = resolve_evaluation_inputs(
            factuals, counterfactuals
        )
        selected_mask = evaluation_mask & success_mask
        selected_counterfactuals = counterfactual_features.loc[selected_mask.to_numpy()]
        if selected_counterfactuals.empty:
            return pd.DataFrame(
                [{f"outlier_rate_k_{neighbor}": float("nan") for neighbor in self._neighbors}]
            )
        results = {}
        for neighbor in self._neighbors:
            model = LocalOutlierFactor(
                n_neighbors=neighbor,
                contamination=0.01,
                novelty=True,
            )
            model.fit(reference.to_numpy(dtype="float32"))
            prediction = model.predict(selected_counterfactuals.to_numpy(dtype="float32"))
            results[f"outlier_rate_k_{neighbor}"] = float(np.mean(prediction == -1))
        return pd.DataFrame([results])


@register(LOCAL_CONNECTEDNESS_EVALUATION_NAME)
class CchvaeConnectednessEvaluation(EvaluationObject):
    def __init__(
        self,
        epsilons: list[int | float] | None = None,
        min_samples: int = DBSCAN_MIN_SAMPLES,
        **kwargs,
    ):
        del kwargs
        self._epsilons = list(epsilons or CONNECTEDNESS_EPSILONS)
        self._min_samples = int(min_samples)

    def evaluate(self, factuals, counterfactuals) -> pd.DataFrame:
        reference = factuals.attr("cchvae_positive_reference")
        _, counterfactual_features, evaluation_mask, success_mask = resolve_evaluation_inputs(
            factuals, counterfactuals
        )
        selected_mask = evaluation_mask & success_mask
        selected_counterfactuals = counterfactual_features.loc[selected_mask.to_numpy()]
        if selected_counterfactuals.empty:
            return pd.DataFrame(
                [
                    {
                        f"not_connected_rate_eps_{epsilon}": float("nan")
                        for epsilon in self._epsilons
                    }
                ]
            )
        reference_array = reference.to_numpy(dtype="float32")
        counter_array = selected_counterfactuals.to_numpy(dtype="float32")
        results = {}
        for epsilon in self._epsilons:
            not_connected = []
            for row in counter_array:
                density_control = np.r_[reference_array, row.reshape(1, -1)]
                labels = (
                    DBSCAN(eps=float(epsilon), min_samples=self._min_samples)
                    .fit(density_control)
                    .labels_
                )
                not_connected.append(labels[-1] == -1)
            results[f"not_connected_rate_eps_{epsilon}"] = float(np.mean(not_connected))
        return pd.DataFrame([results])


def run_reproduction(
    prepare_config_path: Path = PREPARE_CONFIG_PATH,
    run_config_path: Path = RUN_CONFIG_PATH,
    train_sample_limit: int | None = DEFAULT_TRAIN_SAMPLE_LIMIT,
    ncounterfactuals: int | None = DEFAULT_NCOUNTERFACTUALS,
) -> pd.DataFrame:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with prepare_config_path.open("r", encoding="utf-8") as file:
        prepare_config = yaml.safe_load(file)
    if not isinstance(prepare_config, dict):
        raise ValueError("Preparation config must parse to a dictionary")

    with run_config_path.open("r", encoding="utf-8") as file:
        run_config = yaml.safe_load(file)
    if not isinstance(run_config, dict):
        raise ValueError("Run config must parse to a dictionary")

    preparation_experiment = Experiment(prepare_config)
    logger = preparation_experiment._logger
    logger.info("Loaded preparation config from %s", prepare_config_path)
    logger.info("Loaded run config from %s", run_config_path)

    preparation_datasets = [preparation_experiment._raw_dataset]
    for preprocess_step in preparation_experiment._preprocess:
        next_datasets = []
        for current_dataset in preparation_datasets:
            transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                next_datasets.append(transformed)
        preparation_datasets = next_datasets
    trainset, testset = preparation_experiment._resolve_train_test(preparation_datasets)
    logger.info("Train/test sizes: %d / %d", len(trainset), len(testset))

    effective_trainset = trainset
    if train_sample_limit is not None and train_sample_limit < len(trainset):
        limit_df = pd.concat([trainset.get(target=False), trainset.get(target=True)], axis=1)
        effective_trainset = trainset.clone()
        effective_trainset.update(
            "trainset",
            True,
            df=limit_df.iloc[:train_sample_limit].copy(deep=True),
        )
        effective_trainset.freeze()
        TRAIN_LIMIT_PATH.write_text(str(train_sample_limit), encoding="utf-8")
    else:
        TRAIN_LIMIT_PATH.write_text(str(len(trainset)), encoding="utf-8")

    preparation_experiment._target_model.fit(effective_trainset)
    logger.info(
        "Target model trained; best_params=%s",
        getattr(getattr(preparation_experiment._target_model, "_grid_search", None), "best_params_", None),
    )

    desired_class = run_config["method"]["desired_class"]
    class_to_index = preparation_experiment._target_model.get_class_to_index()
    predicted_test = (
        preparation_experiment._target_model.predict(testset, batch_size=512)
        .argmax(dim=1)
        .cpu()
        .numpy()
    )
    desired_index = class_to_index[desired_class]
    search_mask = predicted_test != desired_index
    candidate_indices = testset.get(target=False).index[search_mask]
    selected_indices = (
        candidate_indices
        if ncounterfactuals is None
        else candidate_indices[:ncounterfactuals]
    )
    TEST_INDEX_PATH.write_text(
        json.dumps(selected_indices.tolist()),
        encoding="utf-8",
    )
    logger.info(
        "Selected %d / %d desired-class-mismatched test samples for search",
        len(selected_indices),
        int(search_mask.sum()),
    )

    true_target = trainset.get(target=True).iloc[:, 0]
    true_indices = np.asarray(
        [
            class_to_index[int(value)] if isinstance(value, float) and float(value).is_integer() else class_to_index[value]
            for value in true_target.tolist()
        ],
        dtype=np.int64,
    )
    predicted_train = (
        preparation_experiment._target_model.predict(trainset, batch_size=512)
        .argmax(dim=1)
        .cpu()
        .numpy()
    )
    positive_reference = trainset.get(target=False).loc[
        (predicted_train == desired_index) & (true_indices == desired_index)
    ]
    positive_reference.to_csv(REFERENCE_PATH, index=False)

    experiment = Experiment(run_config)
    metrics = experiment.run()
    results = {
        "dataset": str(run_config["dataset"]["name"]),
        "device": str(run_config["model"]["device"]),
        "n_train": len(trainset),
        "n_test": len(testset),
        "n_search_candidates_total": int(search_mask.sum()),
        "n_counterfactuals_requested": int(len(selected_indices)),
        "n_counterfactuals_evaluated": int(len(selected_indices)),
        "n_counterfactuals_success": (
            int(float(metrics["validity"].iloc[0]) * len(selected_indices))
            if math.isfinite(float(metrics["validity"].iloc[0]))
            else 0
        ),
        **{key: float(value) for key, value in metrics.iloc[0].to_dict().items()},
    }
    result_df = pd.DataFrame([results])

    if not math.isfinite(float(result_df.loc[0, "validity"])) or float(result_df.loc[0, "validity"]) <= 0.0:
        raise AssertionError("validity must be > 0")
    for neighbor in LOF_NEIGHBORS:
        value = float(result_df.loc[0, f"outlier_rate_k_{neighbor}"])
        if math.isfinite(value) and value > OUTLIER_LIMIT:
            raise AssertionError(f"outlier_rate_k_{neighbor}={value:.4f} exceeds limit")
    connectedness_value = float(result_df.loc[0, f"not_connected_rate_eps_{CONNECTEDNESS_EPS}"])
    if math.isfinite(connectedness_value) and connectedness_value > CONNECTEDNESS_LIMIT:
        raise AssertionError(
            f"not_connected_rate_eps_{CONNECTEDNESS_EPS}={connectedness_value:.4f} exceeds limit"
        )

    logger.info("Reproduction metrics:\n%s", result_df.to_string(index=False))
    print(result_df.to_string(index=False))
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-path", default=str(PREPARE_CONFIG_PATH))
    parser.add_argument("--run-path", default=str(RUN_CONFIG_PATH))
    parser.add_argument("--train-sample-limit", type=int, default=None)
    parser.add_argument("--ncounterfactuals", type=int, default=None)
    args = parser.parse_args()
    run_reproduction(
        prepare_config_path=Path(args.prepare_path),
        run_config_path=Path(args.run_path),
        train_sample_limit=args.train_sample_limit,
        ncounterfactuals=args.ncounterfactuals,
    )


if __name__ == "__main__":
    main()
