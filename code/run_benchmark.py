#!/usr/bin/env python3
"""Run the duplicate-aware nested logistic-regression benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metrics_utils import (
    aggregate_probabilities_by_group,
    classification_metrics,
    fit_temperature,
    stable_softmax,
    stratified_group_bootstrap_macro_f1,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "benchmark.json"
OUTPUT = ROOT / "outputs" / "benchmark"
PROBABILITY_COLUMNS = [f"probability_class_{class_id}" for class_id in range(7)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def make_model(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=5000,
                    random_state=seed,
                    multi_class="auto",
                ),
            ),
        ]
    )


def load_data() -> tuple[np.ndarray, pd.DataFrame]:
    with np.load(ROOT / "data" / "source_responses.npz", allow_pickle=False) as archive:
        responses = archive["responses"].astype(np.float32)
    manifest = pd.read_csv(ROOT / "data" / "source_manifest.csv")
    if responses.shape != (350, 150, 8) or len(manifest) != 350:
        raise RuntimeError("unexpected input dimensions")
    features = responses[:, 30:130, :].reshape(350, 800).astype(np.float32)
    return features, manifest


def aligned_probabilities(model: Pipeline, features: np.ndarray) -> np.ndarray:
    probability = model.predict_proba(features)
    classes = model.named_steps["model"].classes_.astype(int)
    output = np.zeros((len(features), 7), dtype=np.float64)
    output[:, classes] = probability
    return np.clip(output, 1e-12, 1.0)


def main() -> int:
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    OUTPUT = parser.parse_args().output.resolve()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    features, manifest = load_data()
    labels = manifest["class_id"].to_numpy(dtype=np.int64)
    groups = manifest["group_id"].astype(str).to_numpy()
    source_ids = manifest["source_id"].astype(str).to_numpy()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    outer = StratifiedGroupKFold(
        n_splits=int(config["outer_folds"]), shuffle=True, random_state=int(config["outer_seed"])
    )
    oof_probability = np.full((len(features), 7), np.nan, dtype=np.float64)
    outer_fold_assignment = np.full(len(features), -1, dtype=np.int64)
    selection_records: list[dict[str, object]] = []
    outer_records: list[dict[str, object]] = []

    for outer_fold, (outer_train, outer_test) in enumerate(outer.split(features, labels, groups), start=1):
        if set(groups[outer_train]) & set(groups[outer_test]):
            raise RuntimeError(f"outer group leakage in fold {outer_fold}")
        inner = StratifiedGroupKFold(
            n_splits=int(config["inner_folds"]),
            shuffle=True,
            random_state=int(config["inner_seed_base"]) + outer_fold,
        )
        candidates: list[dict[str, object]] = []
        for c_value in config["candidate_c"]:
            for probability_state in config["candidate_probability_states"]:
                fold_metrics = []
                for inner_fold, (inner_train_rel, inner_validation_rel) in enumerate(
                    inner.split(features[outer_train], labels[outer_train], groups[outer_train]), start=1
                ):
                    inner_train = outer_train[inner_train_rel]
                    inner_validation = outer_train[inner_validation_rel]
                    if set(groups[inner_train]) & set(groups[inner_validation]):
                        raise RuntimeError(f"inner group leakage in outer {outer_fold}, inner {inner_fold}")
                    model = make_model(float(c_value), int(config["outer_seed"]) + outer_fold * 100 + inner_fold)
                    model.fit(features[inner_train], labels[inner_train])
                    train_probability = aligned_probabilities(model, features[inner_train])
                    validation_probability = aligned_probabilities(model, features[inner_validation])
                    temperature = 1.0
                    if probability_state == "temperature_scaled":
                        temperature = fit_temperature(np.log(train_probability), labels[inner_train])
                        validation_probability = stable_softmax(np.log(validation_probability), temperature)
                    metrics = classification_metrics(labels[inner_validation], validation_probability)
                    fold_metrics.append(
                        {
                            "inner_fold": inner_fold,
                            "temperature": float(temperature),
                            "macro_f1": float(metrics["macro_f1"]),
                            "brier_score": float(metrics["multiclass_brier_score"]),
                            "expected_calibration_error": float(metrics["expected_calibration_error"]),
                        }
                    )
                candidates.append(
                    {
                        "outer_fold": outer_fold,
                        "c": float(c_value),
                        "probability_state": probability_state,
                        "mean_macro_f1": float(np.mean([x["macro_f1"] for x in fold_metrics])),
                        "mean_brier_score": float(np.mean([x["brier_score"] for x in fold_metrics])),
                        "mean_ece": float(np.mean([x["expected_calibration_error"] for x in fold_metrics])),
                        "inner_folds": fold_metrics,
                    }
                )
        best_macro = max(float(x["mean_macro_f1"]) for x in candidates)
        eligible = [x for x in candidates if float(x["mean_macro_f1"]) >= best_macro - float(config["macro_f1_tie_tolerance"])]
        selected = sorted(
            eligible,
            key=lambda x: (
                float(x["mean_brier_score"]),
                float(x["mean_ece"]),
                float(x["c"]),
                str(x["probability_state"]) != "raw",
            ),
        )[0]
        for candidate in candidates:
            selection_records.append({**candidate, "selected": candidate is selected})

        model = make_model(float(selected["c"]), int(config["outer_seed"]) + outer_fold)
        model.fit(features[outer_train], labels[outer_train])
        train_probability = aligned_probabilities(model, features[outer_train])
        test_probability = aligned_probabilities(model, features[outer_test])
        temperature = 1.0
        if selected["probability_state"] == "temperature_scaled":
            temperature = fit_temperature(np.log(train_probability), labels[outer_train])
            test_probability = stable_softmax(np.log(test_probability), temperature)
        oof_probability[outer_test] = test_probability
        outer_fold_assignment[outer_test] = outer_fold
        grouped_test = aggregate_probabilities_by_group(test_probability, labels[outer_test], groups[outer_test])
        fold_metrics = classification_metrics(grouped_test.labels, grouped_test.probabilities)
        outer_records.append(
            {
                "outer_fold": outer_fold,
                "outer_train_sources": int(len(outer_train)),
                "outer_test_sources": int(len(outer_test)),
                "selected_c": float(selected["c"]),
                "selected_probability_state": selected["probability_state"],
                "fitted_temperature": float(temperature),
                **fold_metrics,
            }
        )

    if np.isnan(oof_probability).any() or np.any(outer_fold_assignment < 1):
        raise RuntimeError("incomplete out-of-fold predictions")
    pooled_grouped = aggregate_probabilities_by_group(oof_probability, labels, groups)
    pooled_metrics = classification_metrics(pooled_grouped.labels, pooled_grouped.probabilities)
    bootstrap = stratified_group_bootstrap_macro_f1(
        pooled_grouped.labels,
        pooled_grouped.probabilities,
        resamples=int(config["bootstrap_resamples"]),
        seed=int(config["bootstrap_seed"]),
    )
    thresholds = config["thresholds"]
    decision_rules = {
        "discrimination_supported": bool(
            pooled_metrics["macro_f1"] >= thresholds["macro_f1_min"]
            and bootstrap["ci95_lower"] >= thresholds["macro_f1_ci95_lower_min"]
        ),
        "calibration_supported": bool(
            pooled_metrics["expected_calibration_error"] <= thresholds["ece_max"]
            and pooled_metrics["multiclass_brier_score"] <= thresholds["brier_max"]
        ),
        "ordinal_locality_supported": bool(pooled_metrics["ordinal_step_mae"] <= thresholds["ordinal_step_mae_max"]),
        "all_primary_rules_passed": False,
    }
    decision_rules["all_primary_rules_passed"] = all(
        decision_rules[key]
        for key in ["discrimination_supported", "calibration_supported", "ordinal_locality_supported"]
    )

    predictions = pd.DataFrame(
        {
            "source_id": source_ids,
            "group_id": groups,
            "class_id": labels,
            "outer_fold": outer_fold_assignment,
            "predicted_class_id": oof_probability.argmax(axis=1),
        }
    )
    for class_id, column in enumerate(PROBABILITY_COLUMNS):
        predictions[column] = oof_probability[:, class_id]
    predictions.to_csv(OUTPUT / "nested_oof_predictions.csv", index=False, encoding="utf-8", lineterminator="\n")
    grouped_predictions = pd.DataFrame(
        {
            "group_id": pooled_grouped.group_ids,
            "class_id": pooled_grouped.labels,
            "source_count": pooled_grouped.source_counts,
            "predicted_class_id": pooled_grouped.probabilities.argmax(axis=1),
        }
    )
    for class_id, column in enumerate(PROBABILITY_COLUMNS):
        grouped_predictions[column] = pooled_grouped.probabilities[:, class_id]
    grouped_predictions.to_csv(OUTPUT / "nested_oof_group_predictions.csv", index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame([{k: v for k, v in row.items() if k != "inner_folds"} for row in selection_records]).to_csv(
        OUTPUT / "inner_selection_summary.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    write_json(OUTPUT / "inner_selection_full.json", selection_records)
    write_json(OUTPUT / "outer_fold_metrics.json", outer_records)
    write_json(OUTPUT / "pooled_metrics.json", pooled_metrics)
    write_json(OUTPUT / "pooled_bootstrap.json", bootstrap)
    write_json(OUTPUT / "decision_rules.json", decision_rules)

    print(json.dumps({"metrics": pooled_metrics, "bootstrap": bootstrap}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
