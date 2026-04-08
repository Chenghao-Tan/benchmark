from __future__ import annotations

import argparse
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset.credit.credit import CreditDataset
from method.claproar.claproar import ClaProarMethod
from method.claproar.support import validate_counterfactuals
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_utils import build_optimizer
from preprocess.common import EncodePreProcess, FinalizePreProcess, ScalePreProcess
from utils.seed import seed_context

RANDOM_SEED = 54321
BALANCED_TOTAL = 5000
TRAIN_RATIO = 0.7
N_REPEATS = 5
N_ROUNDS = 50
EVAL_EVERY = 10
RECOURSE_FRACTION = 0.05
RETRAIN_EPOCHS = 10
STATIC_FACTUALS = 5
STATIC_STD_TARGET = 0.03
STATIC_STD_ATOL = 0.2
STATIC_MAX_MEAN_COST = 0.5
MMD_GAMMA = 2.0


class MinimalWachterRunner:
    def __init__(
        self,
        target_model: LinearModel | MlpModel,
        desired_class: int = 1,
        seed: int = RANDOM_SEED,
        lr: float = 0.01,
        lambda_: float = 0.01,
        n_iter: int = 1000,
        tol: float = 1e-4,
    ):
        self._target_model = target_model
        self._desired_class = desired_class
        self._seed = seed
        self._device = target_model._device
        self._lr = float(lr)
        self._lambda = float(lambda_)
        self._n_iter = int(n_iter)
        self._tol = float(tol)
        self._criterion = torch.nn.CrossEntropyLoss()
        self._is_trained = False
        self._feature_names: list[str] = []
        self._desired_index: int | None = None

    def fit(self, trainset) -> None:
        features = trainset.get(target=False)
        try:
            features.to_numpy(dtype="float32")
        except ValueError as error:
            raise ValueError(
                "MinimalWachterRunner requires fully numeric input features"
            ) from error

        class_to_index = self._target_model.get_class_to_index()
        if len(class_to_index) != 2:
            raise ValueError("MinimalWachterRunner supports binary classification only")
        if self._desired_class not in class_to_index:
            raise ValueError("desired_class is invalid for the trained target model")

        self._feature_names = list(features.columns)
        self._desired_index = int(class_to_index[self._desired_class])
        self._is_trained = True

    def get_counterfactuals(
        self, factuals: pd.DataFrame, raw_output: bool = False
    ) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("MinimalWachterRunner is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        try:
            factuals.to_numpy(dtype="float32")
        except ValueError as error:
            raise ValueError(
                "MinimalWachterRunner requires fully numeric input features"
            ) from error

        current_prediction = (
            self._target_model.get_prediction(factuals, proba=True).argmax(dim=1).cpu().numpy()
        )
        if factuals.shape[0] == 0:
            return factuals.copy(deep=True)

        factual_array = factuals.to_numpy(dtype="float32")
        original = torch.tensor(
            factual_array,
            dtype=torch.float32,
            device=self._device,
        )
        candidate = original.clone().detach().requires_grad_(True)
        active_mask = torch.tensor(
            current_prediction != self._desired_index,
            dtype=torch.bool,
            device=self._device,
        )
        target_tensor = torch.full(
            (factuals.shape[0],),
            fill_value=int(self._desired_index),
            dtype=torch.long,
            device=self._device,
        )
        with seed_context(self._seed):
            if bool(active_mask.any()):
                optimizer = torch.optim.Adam([candidate], lr=self._lr)
                for _ in range(self._n_iter):
                    optimizer.zero_grad()
                    logits = self._target_model.forward(candidate)
                    loss = self._criterion(
                        logits[active_mask],
                        target_tensor[active_mask],
                    )
                    distance = torch.linalg.vector_norm(
                        original[active_mask] - candidate[active_mask],
                        ord=1,
                        dim=1,
                    ).mean()
                    objective = loss + self._lambda * distance
                    objective.backward()
                    grad_norm = (
                        None
                        if candidate.grad is None
                        else torch.linalg.vector_norm(candidate.grad[active_mask]).item()
                    )
                    optimizer.step()
                    if grad_norm is not None and grad_norm < self._tol:
                        break

        final_candidates = candidate.detach()
        if bool((~active_mask).any()):
            final_candidates[~active_mask] = original[~active_mask]
        candidates = pd.DataFrame(
            final_candidates.cpu().numpy(),
            index=factuals.index,
            columns=self._feature_names,
        )
        if raw_output:
            return candidates
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("claproar_reproduce")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_frozen_dataset(template, df: pd.DataFrame, marker: str):
    dataset = template.clone()
    dataset.update(marker, True, df=df.copy(deep=True))
    dataset.freeze()
    return dataset


def balance_credit_dataset(
    dataset: CreditDataset,
    total_size: int = BALANCED_TOTAL,
    seed: int = RANDOM_SEED,
) -> CreditDataset:
    target_column = dataset.target_column
    df = dataset.snapshot()
    classes = sorted(pd.Index(df[target_column].unique()).tolist())
    if classes != [0, 1]:
        raise ValueError(f"Expected binary labels [0, 1], received {classes}")

    per_class = total_size // 2
    parts = []
    for label in classes:
        label_df = df.loc[df[target_column] == label]
        if label_df.shape[0] < per_class:
            raise ValueError(
                f"Not enough rows for class {label}: need {per_class}, found {label_df.shape[0]}"
            )
        parts.append(label_df.sample(n=per_class, random_state=seed))

    balanced_df = (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .copy(deep=True)
    )
    dataset.update(
        "balanced_subset",
        {"total_size": int(total_size), "seed": int(seed)},
        df=balanced_df,
    )
    return dataset


def build_credit_workflow(seed: int = RANDOM_SEED):
    dataset = CreditDataset()
    dataset = balance_credit_dataset(dataset, total_size=BALANCED_TOTAL, seed=seed)
    dataset = ScalePreProcess(seed=seed, scaling="standardize", range=True).transform(
        dataset
    )
    dataset = EncodePreProcess(seed=seed, encoding="onehot").transform(dataset)
    trainset, testset = reference_style_split(
        dataset,
        split=1.0 - TRAIN_RATIO,
        seed=seed,
    )
    trainset = FinalizePreProcess(seed=seed).transform(trainset)
    testset = FinalizePreProcess(seed=seed).transform(testset)
    return trainset, testset


def build_credit_static_workflow(seed: int = RANDOM_SEED):
    dataset = CreditDataset()
    dataset = ScalePreProcess(seed=seed, scaling="normalize", range=True).transform(
        dataset
    )
    dataset = EncodePreProcess(seed=seed, encoding="onehot").transform(dataset)
    trainset, testset = reference_style_split(
        dataset,
        split=1.0 - TRAIN_RATIO,
        seed=seed,
    )
    trainset = FinalizePreProcess(seed=seed).transform(trainset)
    testset = FinalizePreProcess(seed=seed).transform(testset)
    return trainset, testset


def reference_style_split(dataset, split: float, seed: int):
    df = dataset.snapshot()
    train_df, test_df = train_test_split(
        df,
        train_size=1.0 - split,
        random_state=seed,
        shuffle=True,
    )
    trainset = dataset
    testset = dataset.clone()
    trainset.update("trainset", True, df=train_df.copy(deep=True))
    testset.update("testset", True, df=test_df.copy(deep=True))
    return trainset, testset


def build_model(model_name: str, device: str) -> LinearModel | MlpModel:
    if model_name == "linear":
        return LinearModel(
            seed=RANDOM_SEED,
            device=device,
            epochs=100,
            learning_rate=0.01,
            batch_size=500,
            optimizer="adam",
            criterion="bce",
            output_activation="sigmoid",
            save_name=None,
        )
    if model_name == "mlp":
        return MlpModel(
            seed=RANDOM_SEED,
            device=device,
            epochs=100,
            learning_rate=0.001,
            batch_size=500,
            layers=[64, 64],
            dropout=0.1,
            optimizer="adam",
            criterion="bce",
            output_activation="sigmoid",
            save_name=None,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def build_static_linear_model(device: str) -> LinearModel:
    return LinearModel(
        seed=RANDOM_SEED,
        device=device,
        epochs=1,
        learning_rate=0.001,
        batch_size=1000,
        optimizer="adam",
        criterion="cross_entropy",
        output_activation="softmax",
        save_name=None,
    )


def continue_training(model: LinearModel | MlpModel, trainset, epochs: int) -> None:
    with seed_context(model._seed):
        X, labels, _ = model.extract_training_data(trainset)
        X_tensor = torch.tensor(
            X.to_numpy(dtype="float32"),
            dtype=torch.float32,
            device=model._device,
        )
        if model._criterion_name == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
            y_tensor = labels.to(model._device)
        else:
            criterion = torch.nn.BCELoss()
            y_tensor = labels.to(model._device).to(dtype=torch.float32).unsqueeze(1)

        optimizer = build_optimizer(
            model._optimizer_name, model._model.parameters(), model._learning_rate
        )

        model._model.train()
        for _ in range(int(epochs)):
            permutation = torch.randperm(X_tensor.shape[0], device=model._device)
            for start in range(0, X_tensor.shape[0], model._batch_size):
                batch_indices = permutation[start : start + model._batch_size]
                batch_X = X_tensor[batch_indices]
                batch_y = y_tensor[batch_indices]
                optimizer.zero_grad()
                logits = model._model(batch_X)
                loss_input = (
                    torch.sigmoid(logits)
                    if model._criterion_name == "bce"
                    else logits
                )
                loss = criterion(loss_input, batch_y)
                loss.backward()
                optimizer.step()

        model._model.eval()
        model._is_trained = True


def select_adverse_factuals(model, dataset, desired_class: int = 1) -> pd.DataFrame:
    feature_df = dataset.get(target=False)
    desired_index = model.get_class_to_index()[desired_class]
    prediction = model.get_prediction(feature_df, proba=True).argmax(dim=1).cpu().numpy()
    return feature_df.loc[prediction != desired_index].copy(deep=True)


def positive_probability_array(
    model: LinearModel | MlpModel, feature_df: pd.DataFrame
) -> np.ndarray:
    probabilities = model.get_prediction(feature_df, proba=True).detach().cpu().numpy()
    class_to_index = model.get_class_to_index()
    positive_index = int(class_to_index[1])
    return probabilities[:, positive_index]


def flatten_model_parameters(model: LinearModel | MlpModel) -> np.ndarray:
    return np.concatenate(
        [parameter.detach().cpu().numpy().reshape(-1) for parameter in model._model.parameters()]
    )


def compute_unbiased_mmd(x: np.ndarray, y: np.ndarray, gamma: float = MMD_GAMMA) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")

    k_xx = rbf_kernel(x, x, gamma=gamma)
    k_yy = rbf_kernel(y, y, gamma=gamma)
    k_xy = rbf_kernel(x, y, gamma=gamma)
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    term_xx = k_xx.sum() / (x.shape[0] * (x.shape[0] - 1))
    term_yy = k_yy.sum() / (y.shape[0] * (y.shape[0] - 1))
    term_xy = 2.0 * k_xy.mean()
    return float(term_xx + term_yy - term_xy)


def compute_positive_class_mmd(
    initial_train_df: pd.DataFrame,
    current_train_df: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
) -> float:
    initial_positive = initial_train_df.loc[
        initial_train_df[target_column] == 1, feature_names
    ].to_numpy(dtype=np.float64)
    current_positive = current_train_df.loc[
        current_train_df[target_column] == 1, feature_names
    ].to_numpy(dtype=np.float64)
    return compute_unbiased_mmd(initial_positive, current_positive)


def compute_pp_mmd(
    initial_model: LinearModel | MlpModel,
    current_model: LinearModel | MlpModel,
    eval_features: pd.DataFrame,
) -> float:
    initial_probabilities = initial_model.get_prediction(eval_features, proba=True).cpu().numpy()
    current_probabilities = current_model.get_prediction(eval_features, proba=True).cpu().numpy()
    return compute_unbiased_mmd(initial_probabilities, current_probabilities)


def compute_disagreement(
    initial_model: LinearModel | MlpModel,
    current_model: LinearModel | MlpModel,
    eval_features: pd.DataFrame,
) -> float:
    initial_prediction = (
        initial_model.get_prediction(eval_features, proba=True).argmax(dim=1).cpu().numpy()
    )
    current_prediction = (
        current_model.get_prediction(eval_features, proba=True).argmax(dim=1).cpu().numpy()
    )
    return float(np.mean(initial_prediction != current_prediction))


def compute_decisiveness(
    model: LinearModel | MlpModel, eval_features: pd.DataFrame
) -> float:
    positive_probability = positive_probability_array(model, eval_features)
    return float(np.mean((positive_probability - 0.5) ** 2))


def compute_f1(
    model: LinearModel | MlpModel, testset
) -> float:
    feature_df = testset.get(target=False)
    target = testset.get(target=True).iloc[:, 0].astype(int).to_numpy()
    positive_probability = positive_probability_array(model, feature_df)
    prediction = (positive_probability >= 0.5).astype(int)
    return float(f1_score(target, prediction))


def build_metrics_row(
    repeat_id: int,
    round_id: int,
    model_name: str,
    method_name: str,
    initial_train_df: pd.DataFrame,
    current_train_df: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
    initial_model: LinearModel | MlpModel,
    current_model: LinearModel | MlpModel,
    eval_features: pd.DataFrame,
    testset,
    initial_f1: float,
    initial_decisiveness: float,
    initial_parameters: np.ndarray,
) -> dict[str, float | int | str]:
    current_parameters = flatten_model_parameters(current_model)
    current_f1 = compute_f1(current_model, testset)
    current_decisiveness = compute_decisiveness(current_model, eval_features)
    return {
        "repeat": int(repeat_id),
        "round": int(round_id),
        "model": model_name,
        "method": method_name,
        "positive_mmd": compute_positive_class_mmd(
            initial_train_df, current_train_df, feature_names, target_column
        ),
        "pp_mmd": compute_pp_mmd(initial_model, current_model, eval_features),
        "disagreement": compute_disagreement(
            initial_model, current_model, eval_features
        ),
        "decisiveness": current_decisiveness,
        "decisiveness_shift": current_decisiveness - initial_decisiveness,
        "f1": current_f1,
        "f1_drop": initial_f1 - current_f1,
        "parameter_shift_l2": float(
            np.linalg.norm(current_parameters - initial_parameters)
        ),
        "positive_count": int((current_train_df[target_column] == 1).sum()),
        "negative_count": int((current_train_df[target_column] == 0).sum()),
    }


def adverse_indices(
    model: LinearModel | MlpModel,
    train_df: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
    desired_class: int = 1,
) -> pd.Index:
    negative_mask = train_df[target_column] != desired_class
    if not bool(negative_mask.any()):
        return pd.Index([])

    factual_features = train_df.loc[negative_mask, feature_names].copy(deep=True)
    desired_index = model.get_class_to_index()[desired_class]
    prediction = (
        model.get_prediction(factual_features, proba=True).argmax(dim=1).cpu().numpy()
    )
    return factual_features.index[prediction != desired_index]


def apply_recourse_batch(
    state: dict,
    batch_indices: pd.Index,
    train_template,
    feature_names: list[str],
    target_column: str,
) -> int:
    current_trainset = build_frozen_dataset(train_template, state["train_df"], "trainset")
    factuals = current_trainset.get(target=False).loc[batch_indices].copy(deep=True)
    counterfactuals = state["method"].get_counterfactuals(factuals)
    success_mask = ~counterfactuals.isna().any(axis=1)
    successful_indices = counterfactuals.index[success_mask]
    if len(successful_indices) > 0:
        state["train_df"].loc[successful_indices, feature_names] = counterfactuals.loc[
            successful_indices, feature_names
        ]
        state["train_df"].loc[successful_indices, target_column] = 1
    return int(len(successful_indices))


def create_workflow_state(
    method_name: str,
    base_model: LinearModel | MlpModel,
    trainset,
    initial_train_df: pd.DataFrame,
) -> dict:
    model = deepcopy(base_model)
    if method_name == "claproar":
        method = ClaProarMethod(
            target_model=model,
            seed=RANDOM_SEED,
            device=model._device,
            desired_class=1,
            individual_cost_lambda=0.1,
            external_cost_lambda=0.1,
            learning_rate=0.01,
            max_iter=100,
            tol=1e-4,
        )
    elif method_name == "wachter":
        method = MinimalWachterRunner(
            target_model=model,
            desired_class=1,
            seed=RANDOM_SEED,
            lr=0.01,
            lambda_=0.01,
            n_iter=1000,
            tol=1e-4,
        )
    else:
        raise ValueError(f"Unsupported method name: {method_name}")

    method.fit(trainset)
    return {
        "method_name": method_name,
        "model": model,
        "initial_model": deepcopy(model),
        "method": method,
        "train_df": initial_train_df.copy(deep=True),
    }


def run_static_reproduction(device: str, output_dir: Path) -> dict[str, float]:
    logger = configure_logger()
    logger.info("Running static ClaPROAR sanity reproduction on credit/linear")

    trainset, testset = build_credit_static_workflow(seed=RANDOM_SEED)
    model = build_static_linear_model(device)
    model.fit(trainset)

    claproar = ClaProarMethod(
        target_model=model,
        seed=RANDOM_SEED,
        device=device,
        desired_class=1,
        individual_cost_lambda=0.1,
        external_cost_lambda=0.1,
        learning_rate=0.01,
        max_iter=100,
        tol=1e-4,
    )
    claproar.fit(trainset)

    adverse_factuals = select_adverse_factuals(model, testset, desired_class=1)
    if adverse_factuals.shape[0] < STATIC_FACTUALS:
        raise ValueError(
            f"Need at least {STATIC_FACTUALS} adverse factuals, found {adverse_factuals.shape[0]}"
        )
    factuals = adverse_factuals.iloc[:STATIC_FACTUALS].copy(deep=True)
    raw_counterfactuals = claproar.get_counterfactuals(factuals, raw_output=True)

    factuals_np = factuals.to_numpy(dtype=np.float64)
    counterfactuals_np = raw_counterfactuals.to_numpy(dtype=np.float64)
    differences = np.abs(counterfactuals_np - factuals_np)
    std_deviation = np.std(differences, axis=0)
    individual_cost = np.linalg.norm(counterfactuals_np - factuals_np, axis=1).mean()

    static_summary = {
        "num_factuals": int(STATIC_FACTUALS),
        "std_mean": float(np.mean(std_deviation)),
        "std_min": float(np.min(std_deviation)),
        "std_max": float(np.max(std_deviation)),
        "individual_cost_mean": float(individual_cost),
    }
    logger.info("Static sanity raw summary: %s", static_summary)

    std_matches_reference = np.allclose(
        std_deviation,
        np.full_like(std_deviation, STATIC_STD_TARGET),
        atol=STATIC_STD_ATOL,
        )
    cost_matches_reference = bool(individual_cost <= STATIC_MAX_MEAN_COST)
    static_summary["std_matches_reference"] = bool(std_matches_reference)
    static_summary["cost_matches_reference"] = bool(cost_matches_reference)

    if not std_matches_reference:
        logger.warning(
            "Static standard-deviation sanity check does not match the reference tolerance"
        )
    if not cost_matches_reference:
        logger.warning(
            "Static individual cost sanity check is higher than the reference tolerance"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([static_summary]).to_csv(
        output_dir / "static_summary.csv", index=False
    )
    logger.info("Static sanity summary: %s", static_summary)
    return static_summary


def run_dynamic_reproduction(
    model_name: str,
    device: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = configure_logger()
    logger.info("Running dynamic ClaPROAR reproduction for model=%s", model_name)

    trainset, testset = build_credit_workflow(seed=RANDOM_SEED)
    feature_names = list(trainset.get(target=False).columns)
    target_column = trainset.target_column
    initial_train_df = pd.concat([trainset.get(target=False), trainset.get(target=True)], axis=1)
    initial_train_df = initial_train_df.astype(
        {feature_name: "float32" for feature_name in feature_names}
        | {target_column: int}
    )
    eval_features = testset.get(target=False).copy(deep=True)
    initial_trainset = build_frozen_dataset(trainset, initial_train_df, "trainset")

    records: list[dict[str, float | int | str]] = []
    repeat_iterator = tqdm(range(N_REPEATS), desc=f"{model_name}-repeats", leave=False)
    for repeat_id in repeat_iterator:
        base_model = build_model(model_name, device)
        base_model.fit(initial_trainset)

        workflows = {
            "wachter": create_workflow_state(
                "wachter", base_model, initial_trainset, initial_train_df
            ),
            "claproar": create_workflow_state(
                "claproar", base_model, initial_trainset, initial_train_df
            ),
        }

        for state in workflows.values():
            initial_f1 = compute_f1(state["initial_model"], testset)
            initial_decisiveness = compute_decisiveness(
                state["initial_model"], eval_features
            )
            initial_parameters = flatten_model_parameters(state["initial_model"])
            records.append(
                build_metrics_row(
                    repeat_id=repeat_id,
                    round_id=0,
                    model_name=model_name,
                    method_name=state["method_name"],
                    initial_train_df=initial_train_df,
                    current_train_df=state["train_df"],
                    feature_names=feature_names,
                    target_column=target_column,
                    initial_model=state["initial_model"],
                    current_model=state["model"],
                    eval_features=eval_features,
                    testset=testset,
                    initial_f1=initial_f1,
                    initial_decisiveness=initial_decisiveness,
                    initial_parameters=initial_parameters,
                )
            )

        round_iterator = tqdm(
            range(1, N_ROUNDS + 1),
            desc=f"{model_name}-repeat-{repeat_id}",
            leave=False,
        )
        for round_id in round_iterator:
            adverse_wachter = adverse_indices(
                workflows["wachter"]["model"],
                workflows["wachter"]["train_df"],
                feature_names,
                target_column,
            )
            adverse_claproar = adverse_indices(
                workflows["claproar"]["model"],
                workflows["claproar"]["train_df"],
                feature_names,
                target_column,
            )
            candidate_pool = adverse_wachter.intersection(adverse_claproar)

            if len(candidate_pool) > 0:
                batch_size = max(1, math.ceil(RECOURSE_FRACTION * len(candidate_pool)))
                rng = np.random.default_rng(RANDOM_SEED + repeat_id + round_id)
                sampled_indices = pd.Index(
                    rng.choice(candidate_pool.to_numpy(), size=batch_size, replace=False)
                )

                for state in workflows.values():
                    apply_recourse_batch(
                        state,
                        sampled_indices,
                        trainset,
                        feature_names,
                        target_column,
                    )

            for state in workflows.values():
                current_trainset = build_frozen_dataset(
                    trainset, state["train_df"], "trainset"
                )
                continue_training(state["model"], current_trainset, RETRAIN_EPOCHS)

            if round_id % EVAL_EVERY != 0 and round_id != N_ROUNDS:
                continue

            for state in workflows.values():
                initial_f1 = compute_f1(state["initial_model"], testset)
                initial_decisiveness = compute_decisiveness(
                    state["initial_model"], eval_features
                )
                initial_parameters = flatten_model_parameters(state["initial_model"])
                records.append(
                    build_metrics_row(
                        repeat_id=repeat_id,
                        round_id=round_id,
                        model_name=model_name,
                        method_name=state["method_name"],
                        initial_train_df=initial_train_df,
                        current_train_df=state["train_df"],
                        feature_names=feature_names,
                        target_column=target_column,
                        initial_model=state["initial_model"],
                        current_model=state["model"],
                        eval_features=eval_features,
                        testset=testset,
                        initial_f1=initial_f1,
                        initial_decisiveness=initial_decisiveness,
                        initial_parameters=initial_parameters,
                    )
                )

    records_df = pd.DataFrame(records)
    summary_df = (
        records_df.groupby(["round", "model", "method"], as_index=False)[
            [
                "positive_mmd",
                "pp_mmd",
                "disagreement",
                "decisiveness",
                "decisiveness_shift",
                "f1",
                "f1_drop",
                "parameter_shift_l2",
                "positive_count",
                "negative_count",
            ]
        ]
        .mean()
    )

    round50 = summary_df.loc[summary_df["round"] == N_ROUNDS].copy(deep=True)
    wachter_row = round50.loc[round50["method"] == "wachter"].iloc[0]
    claproar_row = round50.loc[round50["method"] == "claproar"].iloc[0]
    if float(claproar_row["pp_mmd"]) > float(wachter_row["pp_mmd"]) + 1e-8:
        raise AssertionError(f"ClaPROAR PP-MMD is worse than Wachter for {model_name}")
    if float(claproar_row["disagreement"]) > float(wachter_row["disagreement"]) + 1e-8:
        raise AssertionError(
            f"ClaPROAR disagreement is worse than Wachter for {model_name}"
        )
    if float(claproar_row["f1_drop"]) > float(wachter_row["f1_drop"]) + 1e-8:
        raise AssertionError(f"ClaPROAR F1 drop is worse than Wachter for {model_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records_df.to_csv(output_dir / f"dynamic_{model_name}_records.csv", index=False)
    summary_df.to_csv(output_dir / f"dynamic_{model_name}_summary.csv", index=False)
    logger.info("Dynamic round-50 summary for %s:\n%s", model_name, round50.to_string(index=False))
    return records_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["static", "dynamic", "all"],
        default="all",
    )
    parser.add_argument(
        "--model",
        choices=["linear", "mlp", "all"],
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/claproar/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    device = resolve_device()
    logger = configure_logger()
    logger.info(
        "Starting ClaPROAR reproduction on defaultCredit with device=%s; this does not cover the full paper dataset suite",
        device,
    )

    if args.mode in {"static", "all"}:
        run_static_reproduction(device=device, output_dir=output_dir)

    if args.mode in {"dynamic", "all"}:
        dynamic_models = ["linear", "mlp"] if args.model == "all" else [args.model]
        for model_name in dynamic_models:
            run_dynamic_reproduction(
                model_name=model_name,
                device=device,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    main()
