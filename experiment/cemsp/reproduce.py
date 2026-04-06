from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.distance import cdist, euclidean
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import dataset  # noqa: F401
import method  # noqa: F401
import model  # noqa: F401
import preprocess  # noqa: F401
from experiment import Experiment
from utils.registry import get_registry


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _apply_device(config: dict, device: str) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["device"] = device
    cfg["method"]["device"] = device
    return cfg


def _normalize_config(config: dict) -> dict:
    cfg = deepcopy(config)
    preprocess_cfg = list(cfg.get("preprocess", []))
    if not any(item.get("name", "").lower() == "finalize" for item in preprocess_cfg):
        preprocess_cfg.append({"name": "finalize"})
    cfg["preprocess"] = preprocess_cfg
    return cfg


def _build_dataset(config: dict):
    dataset_cfg = deepcopy(config["dataset"])
    dataset_name = dataset_cfg.pop("name")
    dataset_class = get_registry("dataset")[dataset_name]
    return dataset_class(**dataset_cfg)


def _build_model(config: dict):
    model_cfg = deepcopy(config["model"])
    model_name = model_cfg.pop("name")
    model_class = get_registry("model")[model_name]
    return model_class(**model_cfg)


def _build_method(
    config: dict,
    target_model,
    explicit_lower_bounds: Sequence[float] | None = None,
    explicit_upper_bounds: Sequence[float] | None = None,
):
    method_cfg = deepcopy(config["method"])
    method_name = method_cfg.pop("name")
    if explicit_lower_bounds is not None:
        method_cfg["explicit_lower_bounds"] = list(explicit_lower_bounds)
    if explicit_upper_bounds is not None:
        method_cfg["explicit_upper_bounds"] = list(explicit_upper_bounds)
    method_class = get_registry("method")[method_name]
    return method_class(target_model=target_model, **method_cfg)


def _build_dataset_clone(dataset, features: pd.DataFrame, target: pd.DataFrame, flag: str):
    clone = dataset.clone()
    target_column = dataset.target_column
    combined = pd.concat([features.copy(deep=True), target.copy(deep=True)], axis=1)
    combined = combined.loc[:, [*features.columns, target_column]]
    clone.update(flag, True, df=combined)
    clone.freeze()
    return clone


def _materialize_reference_split(raw_dataset, split_seed: int):
    df = raw_dataset.snapshot()
    target_column = raw_dataset.target_column
    target = df[target_column]
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=split_seed,
        stratify=target,
    )
    train_df = train_df.copy(deep=True)
    test_df = test_df.copy(deep=True)

    trainset_raw = raw_dataset
    testset_raw = raw_dataset.clone()
    trainset_raw.update("trainset", True, df=train_df)
    testset_raw.update("testset", True, df=test_df)
    trainset_raw.freeze()
    testset_raw.freeze()
    return trainset_raw, testset_raw


def _scale_reference_split(trainset_raw, testset_raw):
    scaler = StandardScaler()
    train_x = pd.DataFrame(
        scaler.fit_transform(trainset_raw.get(target=False)),
        index=trainset_raw.get(target=False).index,
        columns=trainset_raw.get(target=False).columns,
    )
    test_x = pd.DataFrame(
        scaler.transform(testset_raw.get(target=False)),
        index=testset_raw.get(target=False).index,
        columns=testset_raw.get(target=False).columns,
    )
    train_y = trainset_raw.get(target=True)
    test_y = testset_raw.get(target=True)
    trainset = _build_dataset_clone(trainset_raw, train_x, train_y, "trainset")
    testset = _build_dataset_clone(testset_raw, test_x, test_y, "testset")
    return trainset, testset, scaler


def _compute_model_metrics(model, testset) -> dict[str, float]:
    probabilities = model.predict_proba(testset).detach().cpu()
    predictions = probabilities.argmax(dim=1)
    target = testset.get(target=True).iloc[:, 0].astype(int).to_numpy()
    accuracy = float(accuracy_score(target, predictions.numpy()))
    f1 = float(f1_score(target, predictions.numpy()))
    return {"test_accuracy": accuracy, "test_f1": f1}


def _hausdorff_score(cfs1, cfs2) -> float:
    cfs1 = np.asarray(cfs1, dtype=np.float64)
    cfs2 = np.asarray(cfs2, dtype=np.float64)
    pairwise_distance = cdist(cfs1, cfs2)
    h_a_b = pairwise_distance.min(1).mean()
    h_b_a = pairwise_distance.min(0).mean()
    return float(max(h_a_b, h_b_a))


def _percent_range(values, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0:
        return array
    start = int(lower * array.shape[0])
    end = int(upper * array.shape[0])
    sliced = array[start:end]
    if sliced.shape[0] == 0:
        return array
    return sliced


class _Evaluator:
    def __init__(self, dataset: np.ndarray, tp_set: np.ndarray):
        self.dataset = dataset
        self.true_positive = tp_set

    def sparsity(self, input_x: np.ndarray, cfs: np.ndarray, precision: int = 3) -> float:
        lens = len(cfs)
        sparsity = 0.0
        for i in range(lens):
            tmp = (input_x - cfs[i]).round(precision)
            sparsity += float((tmp == 0).sum())
        return float(sparsity / (lens * len(cfs[0])))

    def average_percentile_shift(self, input_x: np.ndarray, cfs: np.ndarray) -> float:
        lens = len(cfs)
        shift = np.zeros(input_x.shape[1], dtype=np.float64)
        for i in range(lens):
            for j in range(input_x.shape[1]):
                src_percentile = np.mean(self.dataset[:, j] <= input_x[:, j]) * 100.0
                tgt_percentile = np.mean(self.dataset[:, j] <= cfs[i, j]) * 100.0
                shift[j] += abs(src_percentile - tgt_percentile)
        return float(shift.sum() / (100.0 * input_x.shape[1]))

    def proximity(self, cfs: np.ndarray) -> float:
        lens = len(cfs)
        proximity = 0.0
        for i in range(lens):
            cf = cfs[i:i + 1]
            distance = cdist(cf, self.true_positive).squeeze()
            nearest = int(np.argmin(distance))
            pivot = self.true_positive[nearest]
            pivot_distance = cdist(pivot[np.newaxis, :], self.true_positive).squeeze()
            pivot_distance[nearest] = float("inf")
            pivot_nearest = self.true_positive[int(np.argmin(pivot_distance))]
            proximity += euclidean(cf.squeeze(), pivot) / (
                euclidean(pivot, pivot_nearest) + 1e-6
            )
        return float(proximity / lens)

    def diversity(self, cfs: np.ndarray) -> float:
        k, d = cfs.shape
        if k <= 1:
            return -1.0
        diversity = 0.0
        for i in range(k - 1):
            for j in range(i + 1, k):
                diversity += np.sum(np.abs(cfs[i] - cfs[j]))
        return float(diversity * 2.0 / (k * (k - 1)))

    def count_diversity(self, cfs: np.ndarray, precision: int = 3) -> float:
        k, d = cfs.shape
        if k <= 1:
            return -1.0
        diversity = 0.0
        for i in range(k - 1):
            for j in range(i + 1, k):
                tmp = (cfs[i] - cfs[j]).round(precision)
                diversity += float((tmp != 0).sum())
        return float(diversity * 2.0 / (k * (k - 1) * d))


def _collect_reference_sets(model, trainset, testset):
    train_x = trainset.get(target=False)
    test_x = testset.get(target=False)
    train_y = trainset.get(target=True).iloc[:, 0].astype(int).to_numpy()
    test_y = testset.get(target=True).iloc[:, 0].astype(int).to_numpy()

    pred_test = model.get_prediction(test_x, proba=False).argmax(dim=1).detach().cpu().numpy()
    pred_train = model.get_prediction(train_x, proba=False).argmax(dim=1).detach().cpu().numpy()

    tn_idx = sorted(set(np.where(test_y == 0)[0]).intersection(np.where(pred_test == 0)[0]))
    tp_idx = sorted(set(np.where(train_y == 1)[0]).intersection(np.where(pred_train == 1)[0]))

    abnormal_test = test_x.iloc[tn_idx].to_numpy(dtype=np.float32)
    normal_train = train_x.iloc[tp_idx].to_numpy(dtype=np.float32)
    return abnormal_test, normal_train


def _run_cemsp_for_model(method, model, abnormal_test: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
    counts = []
    cf_sets = []
    for input_x in tqdm(abnormal_test, desc="cemsp-reproduce", leave=False):
        input_row = input_x.reshape(1, -1)
        to_replace = np.where(input_row < lower_bounds, lower_bounds, input_row)
        to_replace = np.where(to_replace > upper_bounds, upper_bounds, to_replace)
        desired_class = 1
        cfs = method._enumerate_counterfactuals(
            input_row.reshape(-1).astype(np.float64),
            desired_class,
            to_replace.reshape(-1).astype(np.float64),
        )
        if not cfs:
            cf_sets.append(np.empty((0, input_row.shape[1]), dtype=np.float64))
            counts.append(0)
            continue
        cf_array = np.asarray(cfs, dtype=np.float64).reshape((-1, input_row.shape[1]))
        cf_sets.append(cf_array)
        counts.append(cf_array.shape[0])
    return counts, cf_sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./experiment/cemsp/config.yml")
    parser.add_argument("--max-factuals", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = _apply_device(_normalize_config(_load_config(config_path)), device)

    raw_dataset = _build_dataset(config)
    split_seed = int(config["reproduction"]["split_seed"])
    trainset_raw, testset_raw = _materialize_reference_split(raw_dataset, split_seed)
    trainset, testset, scaler = _scale_reference_split(trainset_raw, testset_raw)

    model_a = _build_model(config)
    model_a.fit(trainset)
    metrics_a = _compute_model_metrics(model_a, testset)

    model_b_cfg = deepcopy(config)
    model_b_cfg["model"]["seed"] = int(model_b_cfg["model"]["seed"]) + 1
    model_b = _build_model(model_b_cfg)
    model_b.fit(trainset)
    metrics_b = _compute_model_metrics(model_b, testset)

    abnormal_test, normal_train = _collect_reference_sets(model_a, trainset, testset)
    if args.max_factuals is not None:
        abnormal_test = abnormal_test[: int(args.max_factuals)]

    evaluator = _Evaluator(
        trainset.get(target=False).to_numpy(dtype=np.float64),
        normal_train.astype(np.float64),
    )

    normal_range = np.array(
        [
            config["reproduction"]["normal_range"]["lower"],
            config["reproduction"]["normal_range"]["upper"],
        ],
        dtype=np.float64,
    )
    normal_range = scaler.transform(normal_range).astype(np.float64)
    lower_bounds = normal_range[0:1, :]
    upper_bounds = normal_range[1:2, :]

    method = _build_method(
        config,
        model_a,
        explicit_lower_bounds=lower_bounds.reshape(-1),
        explicit_upper_bounds=upper_bounds.reshape(-1),
    )
    method.fit(trainset)
    counts, cf_sets = _run_cemsp_for_model(
        method, model_a, abnormal_test, lower_bounds, upper_bounds
    )

    method_b = _build_method(
        config,
        model_b,
        explicit_lower_bounds=lower_bounds.reshape(-1),
        explicit_upper_bounds=upper_bounds.reshape(-1),
    )
    method_b.fit(trainset)
    _, cf_sets_b = _run_cemsp_for_model(
        method_b, model_b, abnormal_test, lower_bounds, upper_bounds
    )

    sparsity_list = []
    aps_list = []
    proximity_list = []
    diversity_list = []
    diversity2_list = []
    inconsistency_list = []
    count_diversity_list = []
    count_diversity2_list = []

    for input_x, cf_a, cf_b in zip(abnormal_test, cf_sets, cf_sets_b, strict=False):
        if cf_a.shape[0] > 0:
            sparsity_list.append(evaluator.sparsity(input_x.reshape(1, -1), cf_a))
            aps_list.append(evaluator.average_percentile_shift(input_x.reshape(1, -1), cf_a))
            proximity_list.append(evaluator.proximity(cf_a))
            diversity_list.append(evaluator.diversity(cf_a))
            count_diversity_list.append(evaluator.count_diversity(cf_a))
        if cf_b.shape[0] > 0:
            diversity2_list.append(evaluator.diversity(cf_b))
            count_diversity2_list.append(evaluator.count_diversity(cf_b))
        if cf_a.shape[0] > 0 and cf_b.shape[0] > 0:
            inconsistency_list.append(_hausdorff_score(cf_a, cf_b))

    diversity_valid = [value for value in diversity_list if value != -1]
    diversity2_valid = [value for value in diversity2_list if value != -1]
    count_diversity_valid = [value for value in count_diversity_list if value != -1]
    count_diversity2_valid = [value for value in count_diversity2_list if value != -1]
    inconsistency_valid = _percent_range(inconsistency_list)
    sparsity_valid = _percent_range(sparsity_list)
    aps_valid = _percent_range(aps_list)
    proximity_valid = _percent_range(proximity_list)
    diversity_valid = _percent_range(diversity_valid)
    diversity2_valid = _percent_range(diversity2_valid)
    count_diversity_valid = _percent_range(count_diversity_valid)
    count_diversity2_valid = _percent_range(count_diversity2_valid)

    print("CEMSP Hepatitis Reproduction")
    print(f"device: {device}")
    print(f"train_rows: {len(trainset)}")
    print(f"test_rows: {len(testset)}")
    print(f"model_a_test_accuracy: {metrics_a['test_accuracy']:.4f}")
    print(f"model_a_test_f1: {metrics_a['test_f1']:.4f}")
    print(f"model_b_test_accuracy: {metrics_b['test_accuracy']:.4f}")
    print(f"model_b_test_f1: {metrics_b['test_f1']:.4f}")
    print(f"num_abnormal_factuals: {len(abnormal_test)}")
    print(f"num_with_cf: {int(sum(count > 0 for count in counts))}")
    print(f"mean_num_cf: {float(np.mean(counts)) if counts else float('nan'):.4f}")
    print(f"median_num_cf: {float(np.median(counts)) if counts else float('nan'):.4f}")
    print(f"mean_sparsity: {float(np.mean(sparsity_valid)) if len(sparsity_valid) > 0 else float('nan'):.4f}")
    print(f"mean_aps: {float(np.mean(aps_valid)) if len(aps_valid) > 0 else float('nan'):.4f}")
    print(f"mean_proximity: {float(np.mean(proximity_valid)) if len(proximity_valid) > 0 else float('nan'):.4f}")
    print(f"mean_diversity: {float(np.mean(diversity_valid)) if len(diversity_valid) > 0 else float('nan'):.4f}")
    print(f"mean_diversity2: {float(np.mean(diversity2_valid)) if len(diversity2_valid) > 0 else float('nan'):.4f}")
    print(f"mean_inconsistency: {float(np.mean(inconsistency_valid)) if len(inconsistency_valid) > 0 else float('nan'):.4f}")
    print(f"mean_count_diversity: {float(np.mean(count_diversity_valid)) if len(count_diversity_valid) > 0 else float('nan'):.4f}")
    print(f"mean_count_diversity2: {float(np.mean(count_diversity2_valid)) if len(count_diversity2_valid) > 0 else float('nan'):.4f}")


def _normalize_config(config: dict) -> dict:
    cfg = deepcopy(config)
    preprocess_cfg = list(cfg.get("preprocess", []))
    if not any(item.get("name", "").lower() == "finalize" for item in preprocess_cfg):
        preprocess_cfg.append({"name": "finalize"})
    cfg["preprocess"] = preprocess_cfg
    return cfg


if __name__ == "__main__":
    main()
