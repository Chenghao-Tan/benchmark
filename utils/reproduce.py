from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from dataset.dataset_object import DatasetObject
from experiment import Experiment


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must parse to a dictionary")
    return config


def apply_device(config: dict, device: str) -> dict:
    cfg = deepcopy(config)
    if "model" in cfg:
        cfg["model"]["device"] = device
    if "method" in cfg:
        cfg["method"]["device"] = device
    return cfg


def materialize_datasets(experiment: Experiment) -> tuple[DatasetObject, DatasetObject]:
    datasets = [experiment._raw_dataset]
    for preprocess_step in experiment._preprocess:
        next_datasets: list[DatasetObject] = []
        for current_dataset in datasets:
            transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                next_datasets.append(transformed)
        datasets = next_datasets
    return experiment._resolve_train_test(datasets)


def evaluate_counterfactuals(
    experiment: Experiment,
    factuals: DatasetObject,
    counterfactuals: DatasetObject,
) -> pd.DataFrame:
    evaluation_results = [
        evaluation_step.evaluate(factuals, counterfactuals)
        for evaluation_step in experiment._evaluation
    ]
    return pd.concat(evaluation_results, axis=1)


def clone_with_dataframe(
    dataset: DatasetObject,
    df: pd.DataFrame,
    split_flag: str,
) -> DatasetObject:
    output = dataset.clone()
    output.update(split_flag, True, df=df.copy(deep=True))
    output.freeze()
    return output


def select_factuals(
    target_model,
    testset: DatasetObject,
    desired_class: int | str | None = None,
    num_factuals: int | None = None,
    selection: str = "negative_class",
) -> DatasetObject:
    selection = selection.lower()
    feature_df = testset.get(target=False)

    if selection == "all":
        selected_index = feature_df.index
    else:
        prediction = target_model.predict(testset).argmax(dim=1).detach().cpu().numpy()
        class_to_index = target_model.get_class_to_index()
        if desired_class is None:
            if len(class_to_index) != 2:
                raise ValueError(
                    "selection='negative_class' without desired_class requires binary classification"
                )
            desired_index = max(class_to_index.values())
        else:
            desired_index = class_to_index[desired_class]
        mask = prediction != desired_index
        selected_index = feature_df.index[mask]

    if num_factuals is not None:
        selected_index = selected_index[:num_factuals]

    combined = pd.concat([testset.get(target=False), testset.get(target=True)], axis=1)
    selected_df = combined.loc[selected_index].copy(deep=True)
    return clone_with_dataframe(testset, selected_df, "testset")


def _normalize_label(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return value


def compute_model_accuracy(target_model, testset: DatasetObject) -> float:
    class_to_index = target_model.get_class_to_index()
    y_true = testset.get(target=True).iloc[:, 0].map(_normalize_label)
    encoded_y = torch.tensor(
        [class_to_index[value] for value in y_true.tolist()],
        dtype=torch.long,
    )
    prediction = target_model.predict(testset).argmax(dim=1).detach().cpu()
    return float((prediction == encoded_y).to(dtype=torch.float32).mean())
