from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from experiment import Experiment
from utils.registry import register

PAPER_CONFIG_PATH = Path(__file__).with_name("news_popularity_randomforest_gs_reproduce.yaml")
SMOKE_CONFIG_PATH = Path(__file__).with_name(
    "news_popularity_randomforest_gs_reproduce_smoke.yaml"
)
FULLTEST_CONFIG_PATH = Path(__file__).with_name(
    "news_popularity_randomforest_gs_reproduce_fulltest.yaml"
)
LOCAL_L0_EVALUATION_NAME = "gs_l0_sparsity_local"
PAPER_TARGETS = {
    "test_auc": 0.70,
    "max_l0": 17.0,
    "rate_l0_le_9": 0.80,
    "rate_l0_le_5": 0.30,
}
ASSERT_BOUNDS = {
    "test_auc_min": 0.67,
    "test_auc_max": 0.73,
    "rate_l0_le_9_min": 0.75,
    "rate_l0_le_5_min": 0.25,
}


@register(LOCAL_L0_EVALUATION_NAME)
class GsL0SparsityEvaluation(EvaluationObject):
    def __init__(self, thresholds: list[int] | None = None, **kwargs):
        del kwargs
        self._thresholds = list(thresholds or [5, 9])

    def evaluate(self, factuals, counterfactuals) -> pd.DataFrame:
        factual_features, counterfactual_features, evaluation_mask, success_mask = (
            resolve_evaluation_inputs(factuals, counterfactuals)
        )
        selected_mask = evaluation_mask & success_mask
        selected_count = int(selected_mask.sum())
        results: dict[str, float | int] = {
            "num_counterfactuals_evaluated": int(evaluation_mask.sum()),
            "num_counterfactuals_successful": int(success_mask.sum()),
            "num_counterfactuals_selected": selected_count,
        }
        if selected_count == 0:
            results["mean_l0"] = float("nan")
            results["max_l0"] = float("nan")
            for threshold in self._thresholds:
                results[f"rate_l0_le_{threshold}"] = float("nan")
            return pd.DataFrame([results])

        factual_selected = factual_features.loc[selected_mask.to_numpy()]
        counterfactual_selected = counterfactual_features.loc[selected_mask.to_numpy()]
        l0_per_row = factual_selected.ne(counterfactual_selected).sum(axis=1).astype(float)
        results["mean_l0"] = float(l0_per_row.mean())
        results["max_l0"] = float(l0_per_row.max())
        for threshold in self._thresholds:
            results[f"rate_l0_le_{threshold}"] = float((l0_per_row <= threshold).mean())
        return pd.DataFrame([results])


def run_reproduction(
    config_path: str | Path | None = None,
    mode: str = "paper",
) -> dict:
    if mode not in {"paper", "smoke"}:
        raise ValueError("mode must be 'paper' or 'smoke'")

    resolved_config_path = (
        Path(config_path).resolve()
        if config_path is not None
        else (PAPER_CONFIG_PATH if mode == "paper" else SMOKE_CONFIG_PATH)
    )
    resolved_fulltest_path = FULLTEST_CONFIG_PATH.resolve()

    with resolved_config_path.open("r", encoding="utf-8") as file:
        experiment_config = yaml.safe_load(file)
    if not isinstance(experiment_config, dict):
        raise ValueError("Experiment config must parse to a dictionary")

    with resolved_fulltest_path.open("r", encoding="utf-8") as file:
        fulltest_config = yaml.safe_load(file)
    if not isinstance(fulltest_config, dict):
        raise ValueError("Full-test config must parse to a dictionary")

    artifact_experiment = Experiment(deepcopy(experiment_config))
    artifact_datasets = [artifact_experiment._raw_dataset]
    for preprocess_step in artifact_experiment._preprocess:
        next_datasets = []
        for current_dataset in artifact_datasets:
            transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                next_datasets.append(transformed)
        artifact_datasets = next_datasets
    _, factual_testset = artifact_experiment._resolve_train_test(artifact_datasets)

    experiment = Experiment(deepcopy(experiment_config))
    logger = experiment._logger
    start_time = perf_counter()
    metrics = experiment.run()
    elapsed_seconds = perf_counter() - start_time

    fulltest_experiment = Experiment(deepcopy(fulltest_config))
    fulltest_datasets = [fulltest_experiment._raw_dataset]
    for preprocess_step in fulltest_experiment._preprocess:
        next_datasets = []
        for current_dataset in fulltest_datasets:
            transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                next_datasets.append(transformed)
        fulltest_datasets = next_datasets
    fulltest_trainset, fulltest_testset = fulltest_experiment._resolve_train_test(
        fulltest_datasets
    )

    experiment._target_model.fit(fulltest_trainset)
    probabilities = experiment._target_model.predict_proba(fulltest_testset).detach().cpu()
    predictions = probabilities.argmax(dim=1)
    y = fulltest_testset.get(target=True).iloc[:, 0].astype(int)
    class_to_index = experiment._target_model.get_class_to_index()
    encoded_target = torch.tensor(
        [class_to_index[int(value)] for value in y.tolist()],
        dtype=torch.long,
    )
    positive_index = class_to_index.get(1, max(class_to_index.values()))
    test_accuracy = float(
        (predictions == encoded_target).to(dtype=torch.float32).mean()
    )
    test_auc = float(
        roc_auc_score(
            encoded_target.numpy(),
            probabilities[:, positive_index].numpy(),
        )
    )
    logger.info("Test accuracy: %.4f", test_accuracy)
    logger.info("Test AUC: %.4f", test_auc)

    summary = {
        "mode": mode,
        "selected_n_estimators": int(experiment_config["model"]["n_estimators"]),
        "num_factuals": int(len(factual_testset)),
        "search_seconds": float(elapsed_seconds),
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
        **{
            key: (
                float(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else value
            )
            for key, value in metrics.iloc[0].to_dict().items()
        },
    }
    comparison = {
        "paper_targets": PAPER_TARGETS,
        "reproduced": {
            "test_auc": summary["test_auc"],
            "max_l0": summary["max_l0"],
            "rate_l0_le_9": summary["rate_l0_le_9"],
            "rate_l0_le_5": summary["rate_l0_le_5"],
        },
        "notes": {
            "test_auc_scope": "full_test_split",
            "gs_sparsity_scope": f"{len(factual_testset)}_sampled_test_factuals",
        },
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))
    print(json.dumps(comparison, indent=2, sort_keys=True))

    if mode == "paper":
        assert ASSERT_BOUNDS["test_auc_min"] <= float(summary["test_auc"]) <= ASSERT_BOUNDS[
            "test_auc_max"
        ]
        assert float(summary["rate_l0_le_9"]) >= ASSERT_BOUNDS["rate_l0_le_9_min"]
        assert float(summary["rate_l0_le_5"]) >= ASSERT_BOUNDS["rate_l0_le_5_min"]

    return {
        "summary": summary,
        "metrics": metrics,
        "comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        default=None,
        help="Optional path to a GS experiment YAML. Defaults to the mode-specific YAML.",
    )
    parser.add_argument(
        "--mode",
        default="paper",
        choices=["paper", "smoke"],
        help="Choose the static YAML-backed GS study variant.",
    )
    args = parser.parse_args()
    run_reproduction(config_path=args.path, mode=args.mode)


if __name__ == "__main__":
    main()
