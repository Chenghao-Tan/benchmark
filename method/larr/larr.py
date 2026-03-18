from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.larr.library import LARRecourse
from method.larr.support import (
    RecourseModelAdapter,
    ensure_supported_target_model,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("larr")
class LarrMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        lime_seed: int = 0,
        **kwargs,
    ):
        ensure_supported_target_model(
            target_model,
            (LinearModel, MlpModel),
            "LarrMethod",
        )
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class if desired_class is not None else 1
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._lime_seed = int(lime_seed)
        self._coeffs = None
        self._intercepts = None

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        self._method = LARRecourse(
            weights=self._coeffs,
            bias=self._intercepts,
            alpha=self._alpha,
            seed=self._seed,
        )

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for LarrMethod.fit()")

        with seed_context(self._seed):
            self._feature_names = list(trainset.get(target=False).columns)
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )
            self._train_features = trainset.get(target=False).copy(deep=True)
            self._is_trained = True

    def _predict_proba_array(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._adapter.predict_proba(X), dtype=np.float32)

    def _predict_label_array(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._adapter.predict_label_indices(X), dtype=np.int64)

    def _get_linear_coefficients(
        self, factuals: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        weight = self._target_model._model.weight.detach().cpu().numpy()
        bias = self._target_model._model.bias.detach().cpu().numpy()
        coeffs = weight[1] - weight[0]
        intercept = float(bias[1] - bias[0])
        coeffs_matrix = np.vstack([coeffs] * factuals.shape[0])
        intercepts = np.full(factuals.shape[0], intercept, dtype=np.float32)
        return coeffs_matrix, intercepts

    def _get_lime_coefficients(
        self, factuals: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        coeffs = np.zeros(factuals.shape, dtype=np.float32)
        intercepts = []
        lime_exp = self._method.lime_explanation
        X_train = self._train_features.to_numpy(dtype=np.float32)

        for index, (_, row) in enumerate(factuals.iterrows()):
            np.random.seed(self._lime_seed + index)
            coeff, intercept = lime_exp(
                self._predict_proba_array,
                X_train,
                row.to_numpy(dtype=np.float32),
            )
            coeffs[index] = coeff.astype(np.float32, copy=False)
            intercepts.append(float(intercept))
        return coeffs, np.asarray(intercepts, dtype=np.float32)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        if isinstance(self._target_model, LinearModel):
            coeffs, intercepts = self._get_linear_coefficients(factuals)
        else:
            coeffs, intercepts = self._get_lime_coefficients(factuals)

        train_probs = self._predict_proba_array(
            self._train_features.to_numpy(dtype=np.float32)
        )
        train_labels = train_probs.argmax(axis=1)
        recourse_needed_X_train = self._train_features.loc[train_labels == 0].to_numpy(
            dtype=np.float32
        )
        if recourse_needed_X_train.shape[0] == 0:
            raise ValueError(
                "The model did not predict any failures in the training data"
            )

        self._method.choose_lambda(
            recourse_needed_X_train,
            predict_fn=self._predict_label_array,
            X_train=self._train_features.to_numpy(dtype=np.float32),
            predict_proba_fn=self._predict_proba_array,
        )

        rows = []
        with seed_context(self._seed):
            for index, (_, row) in enumerate(factuals.iterrows()):
                cf = self._method.run_method(
                    row.to_numpy(dtype=np.float32).reshape(1, -1),
                    coeffs[index],
                    intercepts[index],
                    self._beta,
                )
                rows.append(cf)

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
