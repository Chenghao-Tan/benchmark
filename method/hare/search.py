from __future__ import annotations

import numpy as np
import pandas as pd

from method.hare.support import ModelAdapter


def boundary_point_search(
    factual: pd.DataFrame,
    candidate: pd.DataFrame,
    model: ModelAdapter,
    target_index: int,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    start = factual.to_numpy(dtype="float32").reshape(-1)
    end = candidate.to_numpy(dtype="float32").reshape(-1)
    if np.isnan(end).any():
        return pd.DataFrame(np.nan, index=factual.index, columns=factual.columns)

    while np.linalg.norm(start - end, ord=2) >= float(epsilon):
        mid = (start + end) / 2.0
        prediction = int(model.predict_label_indices(mid.reshape(1, -1))[0])
        if prediction == int(target_index):
            end = mid
        else:
            start = mid

    return pd.DataFrame(end.reshape(1, -1), index=factual.index, columns=factual.columns)


def calibrate_candidate_set(
    factual: pd.DataFrame,
    candidates: pd.DataFrame,
    model: ModelAdapter,
    target_index: int,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, row in candidates.iterrows():
        row_df = pd.DataFrame(
            row.to_numpy(dtype="float32").reshape(1, -1),
            index=factual.index,
            columns=factual.columns,
        )
        rows.append(
            boundary_point_search(
                factual=factual,
                candidate=row_df,
                model=model,
                target_index=target_index,
                epsilon=epsilon,
            )
        )
    if not rows:
        return candidates.iloc[0:0].copy(deep=True)
    return pd.concat(rows, axis=0, ignore_index=True)
