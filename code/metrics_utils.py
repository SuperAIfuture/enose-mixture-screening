"""Classification metrics, probability aggregation, and stratified group bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score


ORDINAL_POSITION_BY_CLASS = np.asarray([0, 5, 4, 3, 2, 1, 6], dtype=np.int64)


def stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return exponential / exponential.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probabilities = stable_softmax(logits, temperature)
        selected = probabilities[np.arange(labels.size), labels]
        return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())

    result = minimize_scalar(objective, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    if not result.success:
        raise RuntimeError(f"temperature scaling failed: {result.message}")
    return float(np.exp(result.x))


@dataclass(frozen=True)
class GroupedPredictions:
    group_ids: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    source_counts: np.ndarray


def aggregate_probabilities_by_group(
    probabilities: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
) -> GroupedPredictions:
    group_ids = np.asarray(group_ids, dtype=str)
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    ordered_groups = list(dict.fromkeys(group_ids.tolist()))
    output_probabilities: list[np.ndarray] = []
    output_labels: list[int] = []
    source_counts: list[int] = []
    for group_id in ordered_groups:
        mask = group_ids == group_id
        group_labels = np.unique(labels[mask])
        if group_labels.size != 1:
            raise RuntimeError(f"group {group_id} spans labels {group_labels.tolist()}")
        output_labels.append(int(group_labels[0]))
        output_probabilities.append(probabilities[mask].mean(axis=0))
        source_counts.append(int(mask.sum()))
    return GroupedPredictions(
        group_ids=np.asarray(ordered_groups, dtype=str),
        labels=np.asarray(output_labels, dtype=np.int64),
        probabilities=np.stack(output_probabilities),
        source_counts=np.asarray(source_counts, dtype=np.int64),
    )


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not mask.any():
            continue
        accuracy = np.mean(prediction[mask] == labels[mask])
        result += float(mask.mean()) * abs(float(accuracy) - float(confidence[mask].mean()))
    return float(result)


def multiclass_brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return float(np.square(np.asarray(probabilities, dtype=np.float64) - one_hot).sum(axis=1).mean())


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prediction = probabilities.argmax(axis=1).astype(np.int64)
    ordinal_true = ORDINAL_POSITION_BY_CLASS[labels]
    ordinal_prediction = ORDINAL_POSITION_BY_CLASS[prediction]
    ordinal_distance = np.abs(ordinal_prediction - ordinal_true)
    error_mask = prediction != labels
    within_one_error_fraction = float(np.mean(ordinal_distance[error_mask] <= 1)) if error_mask.any() else 1.0
    adjacent_error_fraction = float(np.mean(ordinal_distance[error_mask] == 1)) if error_mask.any() else 0.0
    return {
        "sample_or_group_count": int(labels.size),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "per_class_recall": [
            float(value) for value in recall_score(labels, prediction, labels=np.arange(7), average=None, zero_division=0)
        ],
        "expected_calibration_error": expected_calibration_error(labels, probabilities),
        "multiclass_brier_score": multiclass_brier_score(labels, probabilities),
        "ordinal_step_mae": float(ordinal_distance.mean()),
        "errors_within_one_ordinal_step_fraction": within_one_error_fraction,
        "adjacent_ordinal_error_fraction": adjacent_error_fraction,
        "confusion_matrix": confusion_matrix(labels, prediction, labels=np.arange(7)).astype(int).tolist(),
    }


def stratified_group_bootstrap_macro_f1(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    prediction = np.asarray(probabilities).argmax(axis=1).astype(np.int64)
    class_indices = [np.flatnonzero(labels == class_id) for class_id in range(7)]
    if any(indices.size == 0 for indices in class_indices):
        raise RuntimeError("bootstrap requires at least one immutable group per class")
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for iteration in range(resamples):
        sampled = np.concatenate([rng.choice(indices, size=indices.size, replace=True) for indices in class_indices])
        values[iteration] = f1_score(labels[sampled], prediction[sampled], average="macro", zero_division=0)
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "resamples": int(resamples),
        "seed": int(seed),
        "point_estimate": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_mean": float(values.mean()),
        "bootstrap_standard_deviation": float(values.std(ddof=1)),
    }

