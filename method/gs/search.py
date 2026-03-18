from __future__ import annotations

import numpy as np
import pandas as pd
from numpy import linalg as LA


def hyper_sphere_coordinates(n_search_samples, instance, high, low, p_norm=2):
    delta_instance = np.random.randn(n_search_samples, instance.shape[1])
    dist = np.random.rand(n_search_samples) * (high - low) + low
    norm_p = LA.norm(delta_instance, ord=p_norm, axis=1)
    d_norm = np.divide(dist, norm_p).reshape(-1, 1)
    delta_instance = np.multiply(delta_instance, d_norm)
    return instance + delta_instance, dist


def growing_spheres_search(
    instance,
    keys_mutable,
    keys_immutable,
    continuous_cols,
    binary_cols,
    feature_order,
    model,
    n_search_samples=1000,
    p_norm=2,
    step=0.2,
    max_iter=1000,
):
    keys_correct = feature_order
    keys_mutable_continuous = list(set(keys_mutable) - set(binary_cols))
    keys_mutable_binary = list(set(keys_mutable) - set(continuous_cols))

    instance_immutable_replicated = np.repeat(
        instance[keys_immutable].values.reshape(1, -1), n_search_samples, axis=0
    )
    instance_replicated = np.repeat(
        instance.values.reshape(1, -1), n_search_samples, axis=0
    )
    instance_mutable_replicated_continuous = np.repeat(
        instance[keys_mutable_continuous].values.reshape(1, -1),
        n_search_samples,
        axis=0,
    )

    low = 0
    high = low + step
    count = 0
    counterfactuals_found = False
    candidate_counterfactual_star = np.empty(instance_replicated.shape[1])
    candidate_counterfactual_star[:] = np.nan

    instance_label = np.argmax(model.predict_proba(instance.values.reshape(1, -1)))

    while (not counterfactuals_found) and (count < max_iter):
        count += 1
        candidate_counterfactuals_continuous, _ = hyper_sphere_coordinates(
            n_search_samples,
            instance_mutable_replicated_continuous,
            high,
            low,
            p_norm,
        )
        candidate_counterfactuals_binary = np.random.binomial(
            n=1, p=0.5, size=n_search_samples * len(keys_mutable_binary)
        ).reshape(n_search_samples, -1)

        candidate_counterfactuals = pd.DataFrame(
            np.c_[
                instance_immutable_replicated,
                candidate_counterfactuals_continuous,
                candidate_counterfactuals_binary,
            ]
        )
        candidate_counterfactuals.columns = (
            keys_immutable + keys_mutable_continuous + keys_mutable_binary
        )
        candidate_counterfactuals = candidate_counterfactuals[keys_correct]

        if p_norm == 1:
            distances = np.abs(
                (candidate_counterfactuals.values - instance_replicated)
            ).sum(axis=1)
        elif p_norm == 2:
            distances = np.square(
                (candidate_counterfactuals.values - instance_replicated)
            ).sum(axis=1)
        else:
            raise ValueError("Distance not defined yet")

        y_candidate_logits = model.predict_proba(candidate_counterfactuals.values)
        y_candidate = np.argmax(y_candidate_logits, axis=1)
        indices = np.where(y_candidate != instance_label)
        candidate_counterfactuals = candidate_counterfactuals.values[indices]
        candidate_dist = distances[indices]

        if len(candidate_dist) > 0:
            min_index = np.argmin(candidate_dist)
            candidate_counterfactual_star = candidate_counterfactuals[min_index]
            counterfactuals_found = True

        low = high
        high = low + step

    return feature_selection(
        instance,
        candidate_counterfactual_star,
        model,
        keys_mutable,
        feature_order,
    )


def feature_selection(
    instance, candidate_counterfactual, model, keys_mutable, feature_order
):
    instance = instance.values if isinstance(instance, pd.Series) else instance
    instance_label = np.argmax(model.predict_proba(instance.reshape(1, -1)))
    mutable_indices = [feature_order.index(feature) for feature in keys_mutable]

    for idx in mutable_indices:
        temp_counterfactual = candidate_counterfactual.copy()
        temp_counterfactual[idx] = instance[idx]
        new_label = np.argmax(model.predict_proba(temp_counterfactual.reshape(1, -1)))
        if new_label != instance_label:
            candidate_counterfactual[idx] = instance[idx]

    return candidate_counterfactual
