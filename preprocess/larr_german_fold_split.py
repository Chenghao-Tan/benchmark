from __future__ import annotations

import pandas as pd

from dataset.dataset_object import DatasetObject
from preprocess.preprocess_object import PreProcessObject
from preprocess.preprocess_utils import ensure_flag_absent
from utils.registry import register
from utils.seed import seed_context


@register("larr_german_fold_split")
class LarrGermanFoldSplitPreProcess(PreProcessObject):
    def __init__(
        self,
        seed: int | None = 0,
        fold: int = 0,
        n_folds: int = 5,
        **kwargs,
    ):
        if int(n_folds) < 2:
            raise ValueError("n_folds must be >= 2")
        self._seed = seed
        self._fold = int(fold)
        self._n_folds = int(n_folds)

    def transform(self, input: DatasetObject) -> tuple[DatasetObject, DatasetObject]:
        with seed_context(self._seed):
            ensure_flag_absent(input, "trainset")
            ensure_flag_absent(input, "testset")

            df = input.snapshot()
            shuffled_df = df.sample(frac=1.0, random_state=self._seed).copy(deep=True)
            fold = self._fold % self._n_folds

            chunks = []
            for index in range(self._n_folds):
                start = int(index / self._n_folds * len(shuffled_df))
                end = int((index + 1) / self._n_folds * len(shuffled_df))
                chunks.append(shuffled_df.iloc[start:end].copy(deep=True))

            test_df = chunks.pop(fold)
            train_df = pd.concat(chunks, axis=0).copy(deep=True)

            testset = input.clone()
            trainset = input
            trainset.update("trainset", True, df=train_df)
            testset.update("testset", True, df=test_df)
            return trainset, testset
