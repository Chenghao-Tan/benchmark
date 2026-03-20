from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from method.cfvae.cfvae import _CFVAE


BIN_DIR = PROJECT_ROOT / "method" / "cfvae" / "bin"
DATASET_PATH = PROJECT_ROOT / "dataset" / "adult_cfvae" / "adult_cfvae.csv"


class BlackBox(nn.Module):
    def __init__(self, input_shape: int):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_dim = 10
        self.predict_net = nn.Sequential(
            nn.Linear(self.input_shape, self.hidden_dim),
            nn.Linear(self.hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_net(x)


class DataLoader:
    def __init__(self, dataframe: pd.DataFrame):
        self.data_df = dataframe.copy(deep=True)
        self.continuous_feature_names = ["age", "hours_per_week"]
        self.outcome_name = "income"
        self.categorical_feature_names = [
            name
            for name in self.data_df.columns.tolist()
            if name not in self.continuous_feature_names + [self.outcome_name]
        ]
        self.one_hot_encoded_data = pd.get_dummies(
            self.data_df,
            drop_first=False,
            columns=self.categorical_feature_names,
        )
        self.encoded_feature_names = [
            name
            for name in self.one_hot_encoded_data.columns.tolist()
            if name != self.outcome_name
        ]

    def de_normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy(deep=True)
        for feature_name in self.continuous_feature_names:
            max_value = self.data_df[feature_name].max()
            min_value = self.data_df[feature_name].min()
            result[feature_name] = (
                df[feature_name] * (max_value - min_value)
            ) + min_value
        return result

    def from_dummies(self, data: pd.DataFrame, prefix_sep: str = "_") -> pd.DataFrame:
        out = data.copy(deep=True)
        for feature_name in self.categorical_feature_names:
            cols = [col for col in data.columns if feature_name + prefix_sep in col]
            labels = [col.replace(feature_name + prefix_sep, "") for col in cols]
            out[feature_name] = pd.Categorical(
                np.array(labels)[np.argmax(data[cols].values, axis=1)]
            )
            out.drop(cols, axis=1, inplace=True)
        return out

    def get_decoded_data(self, data: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(data=data, columns=self.encoded_feature_names)
        return self.from_dummies(df)


def target_class_validity(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    sample_sizes: List[int],
) -> List[float]:
    results = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)
        valid_cf_count = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)
            cf_label = torch.argmax(target_model(x_pred), dim=1)
            valid_cf_count += np.sum(test_y.cpu().numpy() != cf_label.cpu().numpy())
        dataset_size = test_x.shape[0]
        valid_cf_count = valid_cf_count / sample_size
        results.append(100 * valid_cf_count / dataset_size)
    return results


def constraint_feasibility_score_age(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    dataloader: DataLoader,
    sample_sizes: List[int],
) -> tuple[List[float], List[float]]:
    results_valid = []
    results_invalid = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)
        valid_change = 0
        invalid_change = 0
        dataset_size = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)
            cf_label = torch.argmax(target_model(x_pred), dim=1)
            x_pred_df = dataloader.de_normalize_data(
                dataloader.get_decoded_data(x_pred.detach().cpu().numpy())
            )
            x_ori_df = dataloader.de_normalize_data(
                dataloader.get_decoded_data(test_x.detach().cpu().numpy())
            )
            age_idx = x_ori_df.columns.get_loc("age")
            for row_index in range(x_ori_df.shape[0]):
                if cf_label[row_index] == 0:
                    continue
                dataset_size += 1
                if x_pred_df.iloc[row_index, age_idx] >= x_ori_df.iloc[row_index, age_idx]:
                    valid_change += 1
                else:
                    invalid_change += 1
        valid_change = valid_change / sample_size
        invalid_change = invalid_change / sample_size
        dataset_size = dataset_size / sample_size
        results_valid.append(100 * valid_change / dataset_size)
        results_invalid.append(100 * invalid_change / dataset_size)
    return results_valid, results_invalid


def constraint_feasibility_score_age_ed(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    dataloader: DataLoader,
    sample_sizes: List[int],
) -> tuple[List[float], List[float]]:
    education_score = {
        "HS-grad": 0,
        "School": 0,
        "Bachelors": 1,
        "Assoc": 1,
        "Some-college": 1,
        "Masters": 2,
        "Prof-school": 2,
        "Doctorate": 3,
    }
    results_valid = []
    results_invalid = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)
        valid_change = 0
        invalid_change = 0
        dataset_size = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)
            cf_label = torch.argmax(target_model(x_pred), dim=1)
            x_pred_df = dataloader.de_normalize_data(
                dataloader.get_decoded_data(x_pred.detach().cpu().numpy())
            )
            x_ori_df = dataloader.de_normalize_data(
                dataloader.get_decoded_data(test_x.detach().cpu().numpy())
            )
            age_idx = x_ori_df.columns.get_loc("age")
            education_idx = x_ori_df.columns.get_loc("education")
            for row_index in range(x_ori_df.shape[0]):
                if cf_label[row_index] == 0:
                    continue
                dataset_size += 1
                current_education = education_score[x_ori_df.iloc[row_index, education_idx]]
                candidate_education = education_score[
                    x_pred_df.iloc[row_index, education_idx]
                ]
                if candidate_education < current_education:
                    invalid_change += 1
                elif candidate_education == current_education:
                    if x_pred_df.iloc[row_index, age_idx] >= x_ori_df.iloc[row_index, age_idx]:
                        valid_change += 1
                    else:
                        invalid_change += 1
                else:
                    if x_pred_df.iloc[row_index, age_idx] > x_ori_df.iloc[row_index, age_idx]:
                        valid_change += 1
                    else:
                        invalid_change += 1
        valid_change = valid_change / sample_size
        invalid_change = invalid_change / sample_size
        dataset_size = dataset_size / sample_size
        results_valid.append(100 * valid_change / dataset_size)
        results_invalid.append(100 * invalid_change / dataset_size)
    return results_valid, results_invalid


def cat_proximity(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    dataloader: DataLoader,
    sample_sizes: List[int],
) -> List[float]:
    results = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)
        diff_count = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)
            x_pred_df = dataloader.de_normalize_data(
                dataloader.get_decoded_data(x_pred.detach().cpu().numpy())
            )
            x_ori_df = dataloader.de_normalize_data(
                dataloader.get_decoded_data(test_x.detach().cpu().numpy())
            )
            for column in dataloader.categorical_feature_names:
                diff_count += np.sum(
                    np.array(x_ori_df[column], dtype=pd.Series)
                    != np.array(x_pred_df[column], dtype=pd.Series)
                )
        dataset_size = test_x.shape[0]
        diff_count = diff_count / sample_size
        results.append(-1 * diff_count / dataset_size)
    return results


def cont_proximity(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    dataloader: DataLoader,
    mad_feature_weights: Dict[str, float],
    sample_sizes: List[int],
) -> List[float]:
    results = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)
        diff_amount = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)
            x_pred_df = dataloader.de_normalize_data(
                dataloader.get_decoded_data(x_pred.detach().cpu().numpy())
            )
            x_ori_df = dataloader.de_normalize_data(
                dataloader.get_decoded_data(test_x.detach().cpu().numpy())
            )
            for column in dataloader.continuous_feature_names:
                diff_amount += (
                    np.sum(np.abs(x_ori_df[column] - x_pred_df[column]))
                    / mad_feature_weights[column]
                )
        dataset_size = test_x.shape[0]
        diff_amount = diff_amount / sample_size
        results.append(-1 * diff_amount / dataset_size)
    return results


def eval_adult(
    methods: Dict[str, Path],
    encoded_size: int,
    target_model: nn.Module,
    val_dataset_np: np.ndarray,
    dataloader: DataLoader,
    mad_feature_weights: Dict[str, float],
    sample_sizes: List[int],
    constraint: str,
    n_test: int,
    device: torch.device,
) -> Dict[str, Dict[str, List[float]]]:
    results: Dict[str, Dict[str, List[float]]] = {}
    val_dataset = torch.tensor(val_dataset_np).to(device)
    target_model.eval().to(device)

    for name, path in methods.items():
        cf_val: Dict[str, List[np.ndarray]] = {}
        cf_vae = _CFVAE(len(dataloader.encoded_feature_names), encoded_size)
        cf_vae.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        cf_vae.eval().to(device)

        test_x = val_dataset.float().to(device)
        test_y = torch.argmax(target_model(test_x), dim=1).to(device)
        filtered_dataset = val_dataset[test_y == 0]

        for _ in range(n_test):
            cf_val.setdefault("target_class_validity", []).append(
                np.array(
                    target_class_validity(
                        cf_vae, target_model, filtered_dataset, sample_sizes
                    ),
                    dtype=float,
                )
            )
            if constraint == "age":
                valid, invalid = constraint_feasibility_score_age(
                    cf_vae,
                    target_model,
                    filtered_dataset,
                    dataloader,
                    sample_sizes,
                )
            else:
                valid, invalid = constraint_feasibility_score_age_ed(
                    cf_vae,
                    target_model,
                    filtered_dataset,
                    dataloader,
                    sample_sizes,
                )
            cf_val.setdefault("constraint_feasibility_score", []).append(
                100 * np.array(valid) / (np.array(valid) + np.array(invalid))
            )
            cf_val.setdefault("cont_proximity", []).append(
                np.array(
                    cont_proximity(
                        cf_vae,
                        target_model,
                        filtered_dataset,
                        dataloader,
                        mad_feature_weights,
                        sample_sizes,
                    ),
                    dtype=float,
                )
            )
            cf_val.setdefault("cat_proximity", []).append(
                np.array(
                    cat_proximity(
                        cf_vae,
                        target_model,
                        filtered_dataset,
                        dataloader,
                        sample_sizes,
                    ),
                    dtype=float,
                )
            )

        results[name] = {
            metric_name: np.mean(np.stack(metric_values, axis=0), axis=0).tolist()
            for metric_name, metric_values in cf_val.items()
        }
    return results


def compare_results(results: Dict, ref: Dict, tolerance: float) -> pd.DataFrame:
    rows = []
    for dataset_name, ref_methods in ref.items():
        dataset_results = results[dataset_name]
        for method_name, ref_metrics in ref_methods.items():
            method_results = dataset_results[method_name]
            for metric_name, ref_value in ref_metrics.items():
                measured = np.array(method_results[metric_name], dtype=float)
                reference = np.array(ref_value, dtype=float)
                max_diff = float(np.max(np.abs(measured - reference)))
                rows.append(
                    {
                        "dataset": dataset_name,
                        "method": method_name,
                        "metric": metric_name,
                        "measured": measured.tolist(),
                        "reference": reference.tolist(),
                        "max_abs_diff": max_diff,
                        "within_tolerance": bool(np.all(np.abs(measured - reference) <= tolerance)),
                    }
                )
    return pd.DataFrame(rows)


REFERENCE_RESULTS = {
    "adult-age": {
        "BaseCVAE": {
            "target_class_validity": [100.0, 100.0, 100.0],
            "constraint_feasibility_score": [56.82554814, 56.93040991, 56.9399428],
            "cont_proximity": [-2.24059021, -2.254801, -2.24498223],
            "cat_proximity": [-3.26024786, -3.26024786, -3.26024786],
        },
        "BaseVAE": {
            "target_class_validity": [100.0, 100.0, 100.0],
            "constraint_feasibility_score": [42.85033365, 43.11248808, 42.99014935],
            "cont_proximity": [-2.66647616, -2.66112855, -2.66558379],
            "cat_proximity": [-3.12011439, -3.12011439, -3.12011439],
        },
        "ModelApprox": {
            "target_class_validity": [99.5900858, 99.5900858, 99.55513187],
            "constraint_feasibility_score": [84.26668079, 83.60122942, 83.58960991],
            "cont_proximity": [-2.73294234, -2.73510212, -2.73455902],
            "cat_proximity": [-3.26167779, -3.26172545, -3.26151891],
        },
        "ExampleBased": {
            "target_class_validity": [99.52335558, 99.52812202, 99.46298062],
            "constraint_feasibility_score": [74.03769743, 74.18632186, 74.21309514],
            "cont_proximity": [-6.80110826, -6.7996033, -6.80052672],
            "cat_proximity": [-3.72411821, -3.7242612, -3.72443597],
        },
    },
    "adult-age-ed": {
        "BaseCVAE": {
            "target_class_validity": [100.0, 100.0, 100.0],
            "constraint_feasibility_score": [57.01620591, 57.20209724, 56.80012711],
            "cont_proximity": [-2.25337728, -2.24797755, -2.24245525],
            "cat_proximity": [-3.26024786, -3.26024786, -3.26024786],
        },
        "BaseVAE": {
            "target_class_validity": [100.0, 100.0, 100.0],
            "constraint_feasibility_score": [42.59294566, 43.06482364, 43.02192564],
            "cont_proximity": [-2.6661057, -2.66556556, -2.6650458],
            "cat_proximity": [-3.12011439, -3.12011439, -3.12011439],
        },
        "ModelApprox": {
            "target_class_validity": [100.0, 100.0, 100.0],
            "constraint_feasibility_score": [79.5042898, 79.32793136, 79.30092151],
            "cont_proximity": [-2.90320554, -2.90264891, -2.90204051],
            "cat_proximity": [-3.22097235, -3.22054337, -3.21916111],
        },
        "ExampleBased": {
            "target_class_validity": [99.93326978, 99.89513823, 99.92691452],
            "constraint_feasibility_score": [66.35181132, 66.50929669, 66.46958292],
            "cont_proximity": [-3.20324281, -3.2077721, -3.20357766],
            "cat_proximity": [-3.5914204, -3.59394662, -3.59634573],
        },
    },
}


def run_cfvae_reproduce(quick: bool = False) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    torch.manual_seed(10000000)
    dataset = pd.read_csv(DATASET_PATH)
    dataloader = DataLoader(dataset.copy())
    vae_test_dataset = np.load(BIN_DIR / "adult-test-set.npy")
    vae_test_dataset = vae_test_dataset[vae_test_dataset[:, -1] == 0, :]
    vae_test_dataset = vae_test_dataset[:, :-1]

    mad_feature_weights = {"age": 10.0, "hours_per_week": 3.0}
    data_size = len(dataloader.encoded_feature_names)
    target_model = BlackBox(data_size)
    target_model.load_state_dict(torch.load(BIN_DIR / "adult-target-model.pth", map_location="cpu"))
    target_model.eval()

    sample_sizes = [1] if quick else [1, 2, 3]
    n_test = 1 if quick else 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    results["adult-age"] = eval_adult(
        methods={
            "BaseCVAE": BIN_DIR / "adult-margin-0.165-validity_reg-42.0-epoch-25-base-gen.pth",
            "BaseVAE": BIN_DIR / "adult-margin-0.369-validity_reg-73.0-ae_reg-2.0-epoch-25-ae-gen.pth",
            "ModelApprox": BIN_DIR / "adult-margin-0.764-constraint-reg-192.0-validity_reg-29.0-epoch-25-unary-gen.pth",
            "ExampleBased": BIN_DIR / "adult-eval-case-0-supervision-limit-100-const-case-0-margin-0.084-oracle_reg-5999.0-validity_reg-159.0-epoch-50-oracle-gen.pth",
        },
        encoded_size=10,
        target_model=target_model,
        val_dataset_np=vae_test_dataset,
        dataloader=dataloader,
        mad_feature_weights=mad_feature_weights,
        sample_sizes=sample_sizes,
        constraint="age",
        n_test=n_test,
        device=device,
    )
    results["adult-age-ed"] = eval_adult(
        methods={
            "BaseCVAE": BIN_DIR / "adult-margin-0.165-validity_reg-42.0-epoch-25-base-gen.pth",
            "BaseVAE": BIN_DIR / "adult-margin-0.369-validity_reg-73.0-ae_reg-2.0-epoch-25-ae-gen.pth",
            "ModelApprox": BIN_DIR / "adult-margin-0.344-constraint-reg-87.0-validity_reg-76.0-epoch-25-unary-ed-gen.pth",
            "ExampleBased": BIN_DIR / "adult-eval-case-0-supervision-limit-100-const-case-1-margin-0.117-oracle_reg-3807.0-validity_reg-175.0-epoch-50-oracle-gen.pth",
        },
        encoded_size=10,
        target_model=target_model,
        val_dataset_np=vae_test_dataset,
        dataloader=dataloader,
        mad_feature_weights=mad_feature_weights,
        sample_sizes=sample_sizes,
        constraint="age-ed",
        n_test=n_test,
        device=device,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    results = run_cfvae_reproduce(quick=args.quick)
    comparison = compare_results(results, REFERENCE_RESULTS, tolerance=args.tolerance)

    for dataset_name, dataset_results in results.items():
        print(f"===== {dataset_name} =====")
        for method_name, metrics in dataset_results.items():
            print(method_name, metrics)
        print()

    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
