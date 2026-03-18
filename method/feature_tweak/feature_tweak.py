from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import sklearn

from dataset.dataset_object import DatasetObject
from method.feature_tweak.support import (
    RecourseModelAdapter,
    ensure_supported_target_model,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from model.randomforest.randomforest import RandomForestModel
from utils.registry import register
from utils.seed import seed_context


def _l2_cost(a, b):
    return np.linalg.norm(a - b, ord=2)


def search_path(tree, class_labels):
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    values = tree.tree_.value

    leaf_nodes = np.where(children_left == -1)[0]
    leaf_values = values[leaf_nodes].reshape(len(leaf_nodes), len(class_labels))
    leaf_classes = np.argmax(leaf_values, axis=-1)
    leaf_nodes = leaf_nodes[np.where(leaf_classes != 0)[0]]

    paths = {}
    for leaf_node in leaf_nodes:
        child_node = leaf_node
        parent_node = -100
        parents_left = []
        parents_right = []
        while parent_node != 0:
            if np.where(children_left == child_node)[0].shape == (0,):
                parent_left = -1
                parent_right = np.where(children_right == child_node)[0][0]
                parent_node = parent_right
            elif np.where(children_right == child_node)[0].shape == (0,):
                parent_right = -1
                parent_left = np.where(children_left == child_node)[0][0]
                parent_node = parent_left
            parents_left.append(parent_left)
            parents_right.append(parent_right)
            child_node = parent_node
        paths[leaf_node] = (parents_left, parents_right)

    path_info = {}
    for leaf_node, (parents_left, parents_right) in paths.items():
        node_ids = []
        inequality_symbols = []
        thresholds = []
        features = []
        for index in range(len(parents_left)):

            def append(node_id):
                node_ids.append(node_id)
                thresholds.append(threshold[node_id])
                features.append(feature[node_id])

            if parents_left[index] != -1:
                node_id = parents_left[index]
                inequality_symbols.append(0)
                append(node_id)
            elif parents_right[index] != -1:
                node_id = parents_right[index]
                inequality_symbols.append(1)
                append(node_id)

        path_info[leaf_node] = {
            "node_id": node_ids,
            "inequality_symbol": inequality_symbols,
            "threshold": thresholds,
            "feature": features,
        }
    return path_info


@register("feature_tweak")
class FeatureTweakMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        eps: float = 0.1,
        **kwargs,
    ):
        ensure_supported_target_model(
            target_model,
            (RandomForestModel,),
            "FeatureTweakMethod",
        )
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class
        self._eps = float(eps)
        self._cost_func = _l2_cost

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._eps <= 0:
            raise ValueError("eps must be > 0")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for FeatureTweakMethod.fit()")

        with seed_context(self._seed):
            self._feature_names = list(trainset.get(target=False).columns)
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )
            self._forest = self._target_model._model
            self._trees = list(self._forest.estimators_)
            self._is_trained = True

    def _esatisfactory_instance(self, x: np.ndarray, path_info):
        esatisfactory = copy.deepcopy(x)
        for index in range(len(path_info["feature"])):
            feature_idx = path_info["feature"][index]
            threshold_value = path_info["threshold"][index]
            inequality_symbol = path_info["inequality_symbol"][index]
            if inequality_symbol == 0:
                esatisfactory[feature_idx] = threshold_value - self._eps
            elif inequality_symbol == 1:
                esatisfactory[feature_idx] = threshold_value + self._eps
        return esatisfactory

    def _as_dataframe(self, x: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame([x], columns=self._feature_names)

    def _forest_predict_label(self, x: np.ndarray) -> int:
        return int(self._forest.predict(self._as_dataframe(x))[0])

    def _feature_tweaking(
        self, x: np.ndarray, class_labels: list[int], cf_label: int
    ) -> np.ndarray:
        x_out = copy.deepcopy(x)
        delta_min = 10**3

        for tree in self._trees:
            estimator_prediction = int(tree.predict(self._as_dataframe(x))[0])
            if (
                self._forest_predict_label(x) == estimator_prediction
                and estimator_prediction != cf_label
            ):
                paths_info = search_path(tree, class_labels)
                for path_key in paths_info:
                    path_info = paths_info[path_key]
                    es_instance = self._esatisfactory_instance(x, path_info)
                    if (
                        int(tree.predict(self._as_dataframe(es_instance))[0])
                        == cf_label
                        and self._cost_func(x, es_instance) < delta_min
                    ):
                        x_out = es_instance
                        delta_min = self._cost_func(x, es_instance)
        return x_out

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        class_labels = [0, 1]
        class_to_index = self._target_model.get_class_to_index()
        if self._desired_class is None:
            cf_label = 1
        else:
            cf_label = int(class_to_index[self._desired_class])

        rows = []
        with seed_context(self._seed):
            for _, row in factuals.iterrows():
                rows.append(
                    self._feature_tweaking(
                        row.to_numpy(dtype="float32"),
                        class_labels,
                        cf_label,
                    )
                )

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
