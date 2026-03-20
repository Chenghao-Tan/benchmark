from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.dataset_object import DatasetObject
from preprocess.preprocess_object import PreProcessObject
from preprocess.preprocess_utils import ensure_flag_absent
from utils.registry import register
from utils.seed import seed_context


@register("rbr_german_future_append")
class RbrGermanFutureAppendPreProcess(PreProcessObject):
    def __init__(
        self,
        seed: int | None = 0,
        path: str = "./dataset/german_modified/",
        sample_fraction: float = 0.2,
        **kwargs,
    ):
        if float(sample_fraction) <= 0 or float(sample_fraction) > 1:
            raise ValueError("sample_fraction must satisfy 0 < sample_fraction <= 1")
        self._seed = seed
        self._path = path
        self._sample_fraction = float(sample_fraction)

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            ensure_flag_absent(input, "rbr_future_appended")
            df = input.snapshot()
            modified_path = Path(self._path)
            if modified_path.is_dir():
                modified_path = modified_path / "german_modified.csv"
            modified_df = pd.read_csv(modified_path)
            modified_df = modified_df.loc[:, df.columns].copy(deep=True)
            sampled_modified = modified_df.sample(
                frac=self._sample_fraction,
                random_state=self._seed,
            ).copy(deep=True)
            combined_df = pd.concat([df, sampled_modified], axis=0, ignore_index=True)
            input.update("rbr_future_appended", True, df=combined_df)
            return input
