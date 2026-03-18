from __future__ import annotations

import copy

import numpy as np
from scipy.sparse import csgraph, csr_matrix
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph


def graph_search(
    data,
    index,
    keys_immutable,
    model,
    n_neighbors=50,
    p_norm=2,
    mode="knn",
    frac=0.4,
    radius=0.25,
):
    data = choose_random_subset(data, frac, index)

    immutable_constraint_matrix1 = np.ones((len(data), len(data)), dtype=float)
    immutable_constraint_matrix2 = np.ones((len(data), len(data)), dtype=float)
    for immutable_index in range(len(keys_immutable)):
        matrix1, matrix2 = build_constraints(data, immutable_index, keys_immutable)
        immutable_constraint_matrix1 *= matrix1
        immutable_constraint_matrix2 *= matrix2

    y_predicted = np.asarray(model.predict_proba(data.values))
    y_predicted = np.argmax(y_predicted, axis=1)
    y_positive_indices = np.where(y_predicted == 1)

    if mode == "knn":
        boundary = 3
        median = n_neighbors
        is_knn = True
    elif mode == "epsilon":
        boundary = 0.10
        median = radius
        is_knn = False
    else:
        raise ValueError("Only possible values for mode are knn and epsilon")

    neighbors_list = [median - boundary, median, median + boundary]

    candidate_counterfactuals = []
    for neighbor_count in neighbors_list:
        neighbor_candidates = find_counterfactuals(
            candidate_counterfactuals,
            data,
            immutable_constraint_matrix1,
            immutable_constraint_matrix2,
            index,
            neighbor_count,
            y_positive_indices,
            is_knn=is_knn,
        )
        candidate_counterfactuals += neighbor_candidates

    candidate_counterfactuals_star = np.array(candidate_counterfactuals)
    if candidate_counterfactuals_star.size == 0:
        candidate_counterfactuals_star = np.empty(data.values.shape[1])
        candidate_counterfactuals_star[:] = np.nan
        return candidate_counterfactuals_star

    if p_norm == 1:
        c_dist = np.abs((data.values[index] - candidate_counterfactuals_star)).sum(
            axis=1
        )
    elif p_norm == 2:
        c_dist = np.square((data.values[index] - candidate_counterfactuals_star)).sum(
            axis=1
        )
    else:
        raise ValueError("Distance not defined yet. Choose p_norm to be 1 or 2")

    min_index = np.argmin(c_dist)
    return candidate_counterfactuals_star[min_index]


def choose_random_subset(data, frac, index):
    number_samples = int(np.rint(frac * data.values.shape[0]))
    number_samples = max(1, min(number_samples, data.values.shape[0] - 1))
    list_to_choose = (
        np.arange(0, index).tolist()
        + np.arange(index + 1, data.values.shape[0]).tolist()
    )
    chosen_indices = np.random.choice(
        list_to_choose,
        replace=False,
        size=number_samples,
    )
    chosen_indices = [index] + chosen_indices.tolist()
    data = data.iloc[chosen_indices]
    return data.sort_index()


def build_constraints(data, i, keys_immutable, epsilon=0.5):
    immutable_constraint_matrix = np.outer(
        data[keys_immutable[i]].values + epsilon,
        data[keys_immutable[i]].values + epsilon,
    )
    immutable_constraint_matrix1 = immutable_constraint_matrix / ((1 + epsilon) ** 2)
    immutable_constraint_matrix1 = ((immutable_constraint_matrix1 == 1) * 1).astype(
        float
    )
    immutable_constraint_matrix2 = immutable_constraint_matrix / (epsilon**2)
    immutable_constraint_matrix2 = ((immutable_constraint_matrix2 == 1) * 1).astype(
        float
    )
    return immutable_constraint_matrix1, immutable_constraint_matrix2


def find_counterfactuals(
    candidates,
    data,
    immutable_constraint_matrix1,
    immutable_constraint_matrix2,
    index,
    n,
    y_positive_indices,
    is_knn,
):
    candidate_counterfactuals_star = copy.deepcopy(candidates)
    graph = build_graph(
        data, immutable_constraint_matrix1, immutable_constraint_matrix2, is_knn, n
    )
    distances, min_distance = shortest_path(graph, index)

    candidate_min_distances = [
        min_distance,
        min_distance + 1,
        min_distance + 2,
        min_distance + 3,
    ]
    min_distance_indices = np.array([0])
    for min_dist in candidate_min_distances:
        min_distance_indices = np.c_[
            min_distance_indices,
            np.array(np.where(distances == min_dist)),
        ]
    min_distance_indices = np.delete(min_distance_indices, 0)
    indices_counterfactuals = np.intersect1d(
        np.array(y_positive_indices),
        np.array(min_distance_indices),
    )
    for counterfactual_index in range(indices_counterfactuals.shape[0]):
        candidate_counterfactuals_star.append(
            data.values[indices_counterfactuals[counterfactual_index]]
        )

    return candidate_counterfactuals_star


def shortest_path(graph, index):
    distances = csgraph.dijkstra(
        csgraph=graph, directed=False, indices=index, return_predecessors=False
    )
    distances[index] = np.inf
    min_distance = distances.min()
    return distances, min_distance


def build_graph(
    data, immutable_constraint_matrix1, immutable_constraint_matrix2, is_knn, n
):
    if is_knn:
        n_neighbors = int(max(1, round(n)))
        n_neighbors = min(n_neighbors, max(1, data.shape[0] - 1))
        graph = kneighbors_graph(data, n_neighbors=n_neighbors, mode="distance")
    else:
        graph = radius_neighbors_graph(data, radius=float(n), mode="distance")
    graph = graph.toarray()
    graph = graph * immutable_constraint_matrix1 * immutable_constraint_matrix2
    return csr_matrix(graph)
