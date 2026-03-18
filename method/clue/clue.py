from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.clue.library import VAE_gauss_cat_net, training, vae_gradient_search
from method.clue.support import (
    RecourseModelAdapter,
    ensure_supported_target_model,
    resolve_feature_groups,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from utils.caching import get_cache_dir
from utils.registry import register
from utils.seed import seed_context


class _ClueDataView:
    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame, target: str):
        self.df_train = df_train
        self.df_test = df_test
        self.target = target


@register("clue")
class ClueMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        train_vae: bool = True,
        width: int = 10,
        depth: int = 3,
        latent_dim: int = 12,
        batch_size: int = 64,
        epochs: int = 10,
        lr: float = 0.001,
        early_stop: int = 10,
        **kwargs,
    ):
        ensure_supported_target_model(
            target_model,
            (LinearModel, MlpModel),
            "ClueMethod",
        )
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = True
        self._is_trained = False
        self._desired_class = desired_class
        self._train_vae = bool(train_vae)
        self._width = int(width)
        self._depth = int(depth)
        self._latent_dim = int(latent_dim)
        self._batch_size = int(batch_size)
        self._epochs = int(epochs)
        self._lr = float(lr)
        self._early_stop = int(early_stop)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ClueMethod.fit()")

        with seed_context(self._seed):
            feature_groups = resolve_feature_groups(trainset)
            self._feature_names = list(feature_groups.feature_names)
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )

            df_train = trainset.get(target=False).copy(deep=True)
            test_size = max(1, int(np.ceil(0.2 * len(df_train))))
            df_test = df_train.iloc[:test_size].copy(deep=True)
            self._data = _ClueDataView(
                df_train=df_train,
                df_test=df_test,
                target=trainset.target_column,
            )
            self._input_dimension = [1] * len(feature_groups.continuous) + [1] * len(
                feature_groups.categorical
            )

            path = Path(get_cache_dir("clue")) / getattr(trainset, "name", "dataset")
            path.mkdir(parents=True, exist_ok=True)
            if self._train_vae:
                x_train = np.float32(
                    self._adapter.get_ordered_features(df_train).values
                )
                x_test = np.float32(self._adapter.get_ordered_features(df_test).values)
                training(
                    x_train,
                    x_test,
                    self._input_dimension,
                    str(path),
                    self._width,
                    self._depth,
                    self._latent_dim,
                    self._batch_size,
                    self._epochs,
                    self._lr,
                    self._early_stop,
                )

            self._vae = VAE_gauss_cat_net(
                self._input_dimension,
                self._width,
                self._depth,
                self._latent_dim,
                pred_sig=False,
                lr=self._lr,
                cuda=False,
                flatten=False,
            )
            self._vae.load(str(path / "theta_best.dat"))
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        rows = []
        with seed_context(self._seed):
            for _, row in factuals.iterrows():
                rows.append(
                    vae_gradient_search(
                        row.to_numpy(dtype="float32"),
                        self._adapter,
                        self._vae,
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
