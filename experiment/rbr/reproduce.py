from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluation_utils import resolve_evaluation_inputs
from experiment import Experiment
from utils.reproduce import (
    apply_device,
    evaluate_counterfactuals,
    load_config,
    materialize_datasets,
    select_factuals,
)


def _apply_quick(config: dict) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["epochs"] = 20
    cfg["method"]["max_iter"] = 100
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-config",
        default=str(Path(__file__).with_name("german_current_reproduce.yaml")),
    )
    parser.add_argument(
        "--future-config",
        default=str(Path(__file__).with_name("german_future_reproduce.yaml")),
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_config = load_config(args.current_config)
    future_config = load_config(args.future_config)
    if args.quick:
        current_config = _apply_quick(current_config)
        future_config = _apply_quick(future_config)
    current_config = apply_device(current_config, device)
    future_config = apply_device(future_config, device)

    current_experiment = Experiment(current_config)
    current_trainset, current_testset = materialize_datasets(current_experiment)
    current_experiment._target_model.fit(current_trainset)
    current_experiment._method.fit(current_trainset)
    factuals = select_factuals(
        current_experiment._target_model,
        current_testset,
        desired_class=1,
        num_factuals=5,
        selection="negative_class",
    )
    counterfactuals = current_experiment._method.predict(factuals)
    current_metrics = evaluate_counterfactuals(
        current_experiment,
        factuals,
        counterfactuals,
    )
    (
        _factual_features,
        current_counterfactual_features,
        evaluation_mask,
        success_mask,
    ) = resolve_evaluation_inputs(factuals, counterfactuals)

    future_validities = []
    for seed in range(5):
        future_seed_config = deepcopy(future_config)
        future_seed_config["model"]["seed"] = seed
        future_seed_config["method"]["seed"] = seed
        for preprocess_cfg in future_seed_config["preprocess"]:
            preprocess_cfg["seed"] = seed

        future_experiment = Experiment(future_seed_config)
        future_trainset, _future_testset = materialize_datasets(future_experiment)
        future_experiment._target_model.fit(future_trainset)
        selected_mask = (evaluation_mask & success_mask).to_numpy()
        denominator = int(evaluation_mask.sum())
        if denominator == 0:
            future_validities.append(float("nan"))
            continue
        if int(selected_mask.sum()) == 0:
            future_validities.append(0.0)
            continue

        expected_columns = list(future_trainset.get(target=False).columns)
        aligned_counterfactuals = current_counterfactual_features.loc[
            selected_mask
        ].reindex(columns=expected_columns, fill_value=0.0)
        prediction = future_experiment._target_model.get_prediction(
            aligned_counterfactuals,
            proba=False,
        )
        desired_index = future_experiment._target_model.get_class_to_index()[1]
        labels = prediction.argmax(dim=1)
        future_validities.append(
            float(
                (labels == desired_index).to(dtype=torch.float32).sum().item()
                / denominator
            )
        )
    feasible_rate = float((evaluation_mask & success_mask).mean())
    result = pd.DataFrame(
        [
            {
                "current_validity": float(current_metrics["validity"].iloc[0]),
                "future_validity": float(sum(future_validities) / len(future_validities)),
                "l1_cost": float(current_metrics["distance_l1"].iloc[0]),
                "feasible_rate": feasible_rate,
            }
        ]
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
