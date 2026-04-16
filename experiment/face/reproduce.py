from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from dataset.synthetic_face.synthetic_face import SyntheticFaceDataset
from evaluation.evaluation_object import EvaluationObject
from experiment import Experiment
from preprocess.preprocess_object import PreProcessObject
from utils.registry import register

CACHE_DIR = PROJECT_ROOT / "cache" / "face"
SUMMARY_CSV_PATH = CACHE_DIR / "synthetic_face_summary.csv"
SUMMARY_JSON_PATH = CACHE_DIR / "synthetic_face_summary.json"
STUDY_CONFIG_PATH = Path(__file__).with_name("face_study.yaml")
LOCAL_PATH_EVALUATION_NAME = "face_path_metrics_local"
LOCAL_QUERY_EVALUATION_NAME = "face_query_summary_local"
LOCAL_QUERY_PREPROCESS_NAME = "face_query_local"


@register(LOCAL_PATH_EVALUATION_NAME)
class FacePathMetricsEvaluation(EvaluationObject):
    def __init__(self, **kwargs):
        del kwargs

    def evaluate(self, factuals, counterfactuals) -> pd.DataFrame:
        del factuals
        evaluation_mask = pd.Series(True, index=counterfactuals.get(target=False).index)
        if hasattr(counterfactuals, "evaluation_filter"):
            evaluation_mask = (
                counterfactuals.attr("evaluation_filter").iloc[:, 0].astype(bool)
            )

        path_found = counterfactuals.attr("face_path_found").iloc[:, 0].astype(bool)
        path_cost = counterfactuals.attr("face_path_cost").iloc[:, 0].astype(float)
        path_hops = counterfactuals.attr("face_path_hops").iloc[:, 0].astype(float)
        endpoint_density = (
            counterfactuals.attr("face_endpoint_density").iloc[:, 0].astype(float)
        )
        endpoint_confidence = (
            counterfactuals.attr("face_endpoint_confidence").iloc[:, 0].astype(float)
        )
        denominator = int(evaluation_mask.sum())
        selected = evaluation_mask & path_found
        return pd.DataFrame(
            [
                {
                    "reachability": (
                        float("nan")
                        if denominator == 0
                        else float(
                            path_found.loc[evaluation_mask.index][evaluation_mask].mean()
                        )
                    ),
                    "path_cost_mean": (
                        float(path_cost.loc[selected].mean())
                        if bool(selected.any())
                        else float("nan")
                    ),
                    "path_hops_mean": (
                        float(path_hops.loc[selected].mean())
                        if bool(selected.any())
                        else float("nan")
                    ),
                    "endpoint_density_mean": (
                        float(endpoint_density.loc[selected].mean())
                        if bool(selected.any())
                        else float("nan")
                    ),
                    "endpoint_confidence_mean": (
                        float(endpoint_confidence.loc[selected].mean())
                        if bool(selected.any())
                        else float("nan")
                    ),
                }
            ]
        )


@register(LOCAL_QUERY_EVALUATION_NAME)
class FaceQuerySummaryEvaluation(EvaluationObject):
    def __init__(self, **kwargs):
        del kwargs

    def evaluate(self, factuals, counterfactuals) -> pd.DataFrame:
        del factuals
        top_paths = counterfactuals.attr("face_top_paths").iloc[0]
        return pd.DataFrame(
            [
                {
                    "query_path_found": bool(
                        counterfactuals.attr("face_path_found").iloc[0, 0]
                    ),
                    "query_top_paths_json": json.dumps(top_paths),
                }
            ]
        )


@register(LOCAL_QUERY_PREPROCESS_NAME)
class FaceQueryPreProcess(PreProcessObject):
    def __init__(
        self,
        x1: float = 0.0,
        x2: float = 0.0,
        y: int = 0,
        seed: int | None = None,
        **kwargs,
    ):
        del kwargs
        self._seed = seed
        self._x1 = float(x1)
        self._x2 = float(x2)
        self._y = int(y)

    def transform(self, input):
        trainset = input.clone()
        testset = input.clone()
        query_df = pd.DataFrame(
            [{"x1": self._x1, "x2": self._x2, input.target_column: self._y}]
        )
        trainset.update("trainset", True, df=input.snapshot())
        testset.update("testset", True, df=query_df)
        return trainset, testset


@register("face_triptych_local")
class FaceTriptychEvaluation(EvaluationObject):
    def __init__(
        self,
        captions: list[str] | None = None,
        output_path: str | None = None,
        **kwargs,
    ):
        del kwargs
        self._captions = list(captions or [])
        self._output_path = None if output_path is None else Path(output_path)

    def evaluate(self, factuals, counterfactuals) -> pd.DataFrame:
        background_df = pd.concat(
            [factuals.get(target=False), factuals.get(target=True)],
            axis=1,
        )
        query_rows = counterfactuals.get(target=False)
        top_paths_payload = counterfactuals.attr("triptych_top_paths")
        output_path = self._output_path
        if output_path is None:
            raise ValueError("FaceTriptychEvaluation requires output_path")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(
            1,
            len(query_rows),
            figsize=(5 * len(query_rows), 5),
            sharex=True,
            sharey=True,
        )
        if len(query_rows) == 1:
            axes = [axes]

        negatives = background_df.loc[background_df["y"] == 0]
        positives = background_df.loc[background_df["y"] == 1]

        for axis, (_, query_row), caption, payload in zip(
            axes,
            query_rows.iterrows(),
            self._captions,
            top_paths_payload.tolist(),
            strict=True,
        ):
            axis.scatter(negatives["x1"], negatives["x2"], c="#d95f02", s=18, alpha=0.65)
            axis.scatter(positives["x1"], positives["x2"], c="#1f77b4", s=18, alpha=0.65)
            top_paths = json.loads(payload)
            if top_paths:
                for rank, path in enumerate(top_paths):
                    xs = [step["x1"] for step in path["trace"]]
                    ys = [step["x2"] for step in path["trace"]]
                    axis.plot(
                        xs,
                        ys,
                        color="#2e8b57",
                        linewidth=2.6 if rank == 0 else 1.5,
                        alpha=max(0.28, 0.9 - 0.15 * rank),
                    )
                endpoint = top_paths[0]["trace"][-1]
                axis.scatter(
                    [endpoint["x1"]],
                    [endpoint["x2"]],
                    c="#c1121f",
                    s=70,
                    marker="o",
                )
            else:
                axis.text(
                    0.03,
                    0.96,
                    "no counterfactual",
                    transform=axis.transAxes,
                    va="top",
                    fontsize=10,
                )
            axis.scatter([query_row["x1"]], [query_row["x2"]], c="#c1121f", s=70, marker="o")
            axis.set_title(caption)
            axis.set_xlabel("x1")
            axis.set_ylabel("x2")

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return pd.DataFrame([{"triptych_written": 1.0}])


def run_reproduction(config_path: str | Path = STUDY_CONFIG_PATH) -> pd.DataFrame:
    resolved_config_path = Path(config_path).resolve()
    with resolved_config_path.open("r", encoding="utf-8") as file:
        study = yaml.safe_load(file)
    if not isinstance(study, dict) or "runs" not in study:
        raise ValueError("FACE study config must define a 'runs' list")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    grouped_query_payload: dict[str, list[dict]] = {}

    for run_cfg in study["runs"]:
        experiment = Experiment(run_cfg["experiment"])
        metrics = experiment.run()
        testset = None
        datasets = [Experiment(deepcopy(run_cfg["experiment"]))._raw_dataset]
        artifact_experiment = Experiment(deepcopy(run_cfg["experiment"]))
        datasets = [artifact_experiment._raw_dataset]
        for preprocess_step in artifact_experiment._preprocess:
            next_datasets = []
            for current_dataset in datasets:
                transformed = preprocess_step.transform(current_dataset)
                if isinstance(transformed, tuple):
                    next_datasets.extend(list(transformed))
                else:
                    next_datasets.append(transformed)
            datasets = next_datasets
        _, testset = artifact_experiment._resolve_train_test(datasets)
        prediction = experiment._target_model.predict(testset).argmax(dim=1).detach().cpu().numpy()
        target = testset.get(target=True).iloc[:, 0].astype(int).to_numpy()
        class_to_index = experiment._target_model.get_class_to_index()
        encoded_target = [class_to_index[int(value)] for value in target.tolist()]
        model_accuracy = float((prediction == encoded_target).mean())

        summary_rows.append(
            {
                "label": run_cfg["label"],
                "mode": run_cfg["mode"],
                "epsilon": float(run_cfg["epsilon"]),
                "k_neighbors": (
                    float("nan") if run_cfg.get("k_neighbors") is None else int(run_cfg["k_neighbors"])
                ),
                "prediction_threshold": float(run_cfg["prediction_threshold"]),
                "density_threshold": (
                    float("nan")
                    if run_cfg.get("density_threshold") is None
                    else float(run_cfg["density_threshold"])
                ),
                "train_accuracy": model_accuracy,
                **{key: float(value) for key, value in metrics.iloc[0].to_dict().items()},
            }
        )

        query_experiment = Experiment(run_cfg["query_experiment"])
        query_metrics = query_experiment.run()
        grouped_query_payload.setdefault(run_cfg["group"], []).append(
            {
                "caption": run_cfg["caption"],
                "query_top_paths_json": query_metrics.loc[0, "query_top_paths_json"],
                "query_path_found": bool(query_metrics.loc[0, "query_path_found"]),
                "query_point": run_cfg["query_point"],
                "triptych_output_path": run_cfg["triptych_output_path"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mode", "epsilon", "k_neighbors"], na_position="last"
    ).reset_index(drop=True)

    background_dataset = SyntheticFaceDataset()
    background_dataset.update("testset", True, df=background_dataset.snapshot())
    background_dataset.freeze()
    for group, payload in grouped_query_payload.items():
        background_counterfactual = background_dataset.clone()
        query_df = pd.DataFrame(
            [
                {
                    "x1": float(item["query_point"]["x1"]),
                    "x2": float(item["query_point"]["x2"]),
                    background_dataset.target_column: int(item["query_point"]["y"]),
                }
                for item in payload
            ]
        )
        background_counterfactual.update("counterfactual", True, df=query_df)
        background_counterfactual.update(
            "triptych_top_paths",
            pd.Series(
                [item["query_top_paths_json"] for item in payload],
                index=query_df.index,
                name="triptych_top_paths",
                dtype=object,
            ),
        )
        background_counterfactual.freeze()
        FaceTriptychEvaluation(
            captions=[item["caption"] for item in payload],
            output_path=payload[0]["triptych_output_path"],
        ).evaluate(background_dataset, background_counterfactual)

    assert bool((summary_df["train_accuracy"] >= 0.95).all())
    assert float(
        summary_df.loc[summary_df["label"] == "kde_eps_0.50", "reachability"].iloc[0]
    ) > 0.0
    assert float(
        summary_df.loc[summary_df["label"] == "epsilon_eps_0.50", "reachability"].iloc[0]
    ) > 0.0
    assert float(
        summary_df.loc[summary_df["label"] == "epsilon_eps_1.00", "reachability"].iloc[0]
    ) > 0.0
    assert float(
        summary_df.loc[summary_df["label"] == "knn_k_4_eps_0.35", "reachability"].iloc[0]
    ) > 0.0
    knn_rows = summary_df.loc[summary_df["mode"] == "knn"].set_index("k_neighbors")
    assert knn_rows.loc[2.0, "reachability"] <= knn_rows.loc[4.0, "reachability"]
    assert knn_rows.loc[4.0, "reachability"] <= knn_rows.loc[10.0, "reachability"]

    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    SUMMARY_JSON_PATH.write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    return summary_df


def main() -> None:
    summary_df = run_reproduction()
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
