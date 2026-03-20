from __future__ import annotations

import pandas as pd

from dataset.dataset_object import DatasetObject
from preprocess.preprocess_object import PreProcessObject
from preprocess.preprocess_utils import ensure_flag_absent, resolve_feature_metadata
from utils.registry import register
from utils.seed import seed_context


@register("cfvae_adult_prepare")
class CfvaeAdultPreparePreProcess(PreProcessObject):
    def __init__(self, seed: int | None = None, **kwargs):
        self._seed = seed

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            ensure_flag_absent(input, "cfvae_raw_features")
            df = input.snapshot()
            feature_df = df.loc[:, df.columns != input.target_column].copy(deep=True)
            feature_type, _, _ = resolve_feature_metadata(input)

            numeric_bounds: dict[str, dict[str, float]] = {}
            numeric_mad: dict[str, float] = {}
            for feature_name, feature_kind in feature_type.items():
                if feature_name == input.target_column:
                    continue
                if str(feature_kind).lower() != "numerical":
                    continue
                series = pd.to_numeric(feature_df[feature_name], errors="coerce")
                numeric_bounds[feature_name] = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
                median_value = float(series.median())
                mad_value = float((series - median_value).abs().median())
                numeric_mad[feature_name] = mad_value if mad_value > 0 else 1.0

            input.update("cfvae_raw_features", feature_df)
            input.update("cfvae_numeric_bounds", numeric_bounds)
            input.update("cfvae_numeric_mad", numeric_mad)
            return input
