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

from experiment import Experiment
from utils.reproduce import (
    apply_device,
    clone_with_dataframe,
    evaluate_counterfactuals,
    load_config,
    materialize_datasets,
    select_factuals,
)


def _apply_quick(config: dict) -> dict:
    cfg = deepcopy(config)
    cfg["evaluation"][2]["n_samples"] = 500
    cfg["method"]["n_iter"] = 100
    cfg["model"]["epochs"] = 10
    return cfg


def _run_single(config_path: str, quick: bool) -> pd.DataFrame:
    config = load_config(config_path)
    if quick:
        config = _apply_quick(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = apply_device(config, device)

    experiment = Experiment(config)
    trainset, testset = materialize_datasets(experiment)
    experiment._target_model.fit(trainset)
    experiment._method.fit(trainset)

    combined = pd.concat(
        [
            pd.concat([trainset.get(target=False), trainset.get(target=True)], axis=1),
            pd.concat([testset.get(target=False), testset.get(target=True)], axis=1),
        ],
        axis=0,
    )
    factual_dataset = clone_with_dataframe(testset, combined, "testset")
    factuals = select_factuals(
        experiment._target_model,
        factual_dataset,
        desired_class=1,
        num_factuals=5,
        selection="negative_class",
    )
    counterfactuals = experiment._method.predict(factuals)
    metrics = evaluate_counterfactuals(experiment, factuals, counterfactuals)

    return pd.DataFrame(
        [
            {
                "config": Path(config_path).name,
                "average_recourse_accuracy": float(metrics["validity"].iloc[0]),
                "average_invalidation_rate": float(
                    metrics["average_invalidation_rate"].iloc[0]
                ),
                "average_cost": float(metrics["distance_l1"].iloc[0]),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--linear-config",
        default=str(Path(__file__).with_name("compas_carla_linear_reproduce.yaml")),
    )
    parser.add_argument(
        "--mlp-config",
        default=str(Path(__file__).with_name("compas_carla_mlp_reproduce.yaml")),
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    outputs = [
        _run_single(args.linear_config, args.quick),
        _run_single(args.mlp_config, args.quick),
    ]
    print(pd.concat(outputs, axis=0, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
