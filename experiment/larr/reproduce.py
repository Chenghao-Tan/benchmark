from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment import Experiment
from method.larr.library import RecourseCost
from utils.reproduce import apply_device, load_config, materialize_datasets, select_factuals


def _apply_quick(config: dict) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["epochs"] = 20
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("german_mlp_reproduce.yaml")),
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    base_config = load_config(args.config)
    if args.quick:
        base_config = _apply_quick(base_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_config = apply_device(base_config, device)

    robustness_values: list[float] = []
    consistency_values: list[float] = []

    for fold in range(5):
        config = deepcopy(base_config)
        for preprocess_cfg in config["preprocess"]:
            if preprocess_cfg["name"] == "larr_german_fold_split":
                preprocess_cfg["fold"] = fold

        experiment = Experiment(config)
        trainset, testset = materialize_datasets(experiment)
        experiment._target_model.fit(trainset)
        experiment._method.fit(trainset)

        factuals = select_factuals(
            experiment._target_model,
            testset,
            desired_class=1,
            num_factuals=5,
            selection="negative_class",
        )
        factual_df = factuals.get(target=False)
        if factual_df.empty:
            continue

        method = experiment._method
        train_probs = method._predict_proba_array(
            method._train_features.to_numpy(dtype=np.float32)
        )
        train_labels = train_probs.argmax(axis=1)
        recourse_needed_X_train = method._train_features.loc[train_labels == 0].to_numpy(
            dtype=np.float32
        )
        method._method.choose_lambda(
            recourse_needed_X_train,
            predict_fn=method._predict_label_array,
            X_train=method._train_features.to_numpy(dtype=np.float32),
            predict_proba_fn=method._predict_proba_array,
        )

        coeffs, intercepts = method._get_lime_coefficients(factual_df)
        lar_recourse = method._method
        beta = method._beta

        for index, (_, row) in enumerate(factual_df.iterrows()):
            x_0 = row.to_numpy(dtype=np.float32)
            J = RecourseCost(x_0, lar_recourse.lamb)

            lar_recourse.weights = coeffs[index]
            lar_recourse.bias = intercepts[index]

            x_r = lar_recourse.get_recourse(x_0, beta=1.0)
            weights_r, bias_r = lar_recourse.calc_theta_adv(x_r)
            J_r_opt = J.eval(x_r, weights_r, bias_r)

            theta_p = (np.array(weights_r, copy=True), np.array(bias_r, copy=True))
            x_c = lar_recourse.get_recourse(x_0, beta=0.0, theta_p=theta_p)
            J_c_opt = J.eval(x_c, *theta_p)

            x = lar_recourse.get_recourse(x_0, beta=beta, theta_p=theta_p)
            weights_r_beta, bias_r_beta = lar_recourse.calc_theta_adv(x)
            J_r = J.eval(x, weights_r_beta, bias_r_beta)
            J_c = J.eval(x, *theta_p)

            robustness_values.append(float(np.ravel(J_r - J_r_opt)[0]))
            consistency_values.append(float(np.ravel(J_c - J_c_opt)[0]))

    result = pd.DataFrame(
        [
            {
                "average_robustness": float(np.mean(robustness_values)),
                "average_consistency": float(np.mean(consistency_values)),
            }
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "metric": "average_robustness",
                "measured": float(result["average_robustness"].iloc[0]),
                "reference": 0.28,
            },
            {
                "metric": "average_consistency",
                "measured": float(result["average_consistency"].iloc[0]),
                "reference": 0.405,
            },
        ]
    )
    print(result.to_string(index=False))
    print()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
