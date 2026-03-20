from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluation_utils import resolve_evaluation_inputs
from experiment import Experiment
from utils.reproduce import (
    apply_device,
    compute_model_accuracy,
    evaluate_counterfactuals,
    load_config,
    materialize_datasets,
)


REFERENCE_RESULTS = {
    "accuracy": 0.86,
    "validity": 0.9859,
    "sparsity_cat_l0": 0.11,
    "sparsity_num_l1": 0.19,
    "immutability_violation_rate": 0.0,
}


def _apply_quick(config: dict) -> dict:
    cfg = deepcopy(config)
    cfg["method"]["autoencoder_target_steps"] = 200
    cfg["method"]["train_steps"] = 400
    return cfg


def _rbf_mmd(left: np.ndarray, right: np.ndarray, gamma: float | None = None) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    if gamma is None:
        gamma = 1.0 / max(1, left.shape[1])
    xx = np.exp(-gamma * np.square(left[:, None, :] - left[None, :, :]).sum(axis=2))
    yy = np.exp(-gamma * np.square(right[:, None, :] - right[None, :, :]).sum(axis=2))
    xy = np.exp(-gamma * np.square(left[:, None, :] - right[None, :, :]).sum(axis=2))
    return float(xx.mean() + yy.mean() - 2.0 * xy.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("adult_cfrl_randomforest_reproduce.yaml")),
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.quick:
        config = _apply_quick(config)
    config = apply_device(config, "cpu")

    experiment = Experiment(config)
    trainset, testset = materialize_datasets(experiment)
    experiment._target_model.fit(trainset)
    experiment._method.fit(trainset)
    counterfactuals = experiment._method.predict(testset)
    metrics = evaluate_counterfactuals(experiment, testset, counterfactuals)

    (
        _factual_features,
        counterfactual_features,
        evaluation_mask,
        success_mask,
    ) = resolve_evaluation_inputs(testset, counterfactuals)
    selected_mask = (evaluation_mask & success_mask).to_numpy()
    successful_counterfactuals = counterfactual_features.loc[selected_mask].copy(deep=True)

    train_features = trainset.get(target=False)
    train_predictions = (
        experiment._target_model.get_prediction(train_features, proba=False)
        .argmax(dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    desired_index = experiment._target_model.get_class_to_index()[1]
    reference_train = train_features.loc[train_predictions == desired_index].to_numpy(
        dtype=np.float32
    )

    measured = {
        "accuracy": compute_model_accuracy(experiment._target_model, testset),
        "validity": float(metrics["validity"].iloc[0]),
        "sparsity_cat_l0": float(metrics["sparsity_cat_l0"].iloc[0]),
        "sparsity_num_l1": float(metrics["sparsity_num_l1"].iloc[0]),
        "immutability_violation_rate": float(
            metrics["immutability_violation_rate"].iloc[0]
        ),
        "target_conditional_mmd": _rbf_mmd(
            successful_counterfactuals.to_numpy(dtype=np.float32),
            reference_train,
        ),
    }
    comparison = pd.DataFrame(
        [
            {
                "metric": metric,
                "measured": value,
                "reference": REFERENCE_RESULTS.get(metric, float("nan")),
            }
            for metric, value in measured.items()
        ]
    )
    print(metrics.to_string(index=False))
    print()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
