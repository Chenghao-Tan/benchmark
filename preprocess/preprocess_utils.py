"""Shared helpers for preprocessing modules."""

from __future__ import annotations

from dataset.dataset_object import DatasetObject


def dataset_has_attr(dataset: DatasetObject, flag: str) -> bool:
    """Return whether a dataset exposes a given public attribute.

    Args:
        dataset: Dataset to inspect.
        flag: Attribute name to query.

    Returns:
        bool: ``True`` when the dataset has the attribute.
    """
    try:
        dataset.attr(flag)
    except AttributeError:
        return False
    else:
        return True


def ensure_flag_absent(dataset: DatasetObject, flag: str) -> None:
    """Ensure a preprocess marker has not already been attached.

    Args:
        dataset: Dataset to inspect.
        flag: Attribute name reserved for a preprocess step.

    Raises:
        ValueError: If the dataset already contains the requested flag.
    """
    if dataset_has_attr(dataset, flag):
        raise ValueError(f"Dataset already contains preprocess flag: {flag}")


def resolve_feature_metadata(
    dataset: DatasetObject,
) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
    """Return the active feature metadata for a dataset.

    Encoded metadata is preferred when available so downstream preprocessors can
    work on the transformed feature space. Otherwise the raw dataset metadata is
    returned.

    Args:
        dataset: Dataset whose feature metadata should be resolved.

    Returns:
        tuple[dict[str, str], dict[str, bool], dict[str, str]]: Feature type,
        mutability, and actionability mappings.
    """
    if dataset_has_attr(dataset, "encoded_feature_type"):
        feature_type = dataset.attr("encoded_feature_type")
        feature_mutability = dataset.attr("encoded_feature_mutability")
        feature_actionability = dataset.attr("encoded_feature_actionability")
    else:
        feature_type = dataset.attr("raw_feature_type")
        feature_mutability = dataset.attr("raw_feature_mutability")
        feature_actionability = dataset.attr("raw_feature_actionability")
    return feature_type, feature_mutability, feature_actionability
