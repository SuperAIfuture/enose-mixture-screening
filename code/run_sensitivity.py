"""Nested time-window, channel-removal, and duplicate-removal analyses."""
from __future__ import annotations
import argparse, json, os, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name,"1")
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from metrics_utils import aggregate_probabilities_by_group, classification_metrics, fit_temperature, stable_softmax
ROOT = Path(__file__).resolve().parents[1]
NESTED_ROOT = ROOT / "outputs" / "sensitivity"
PROBABILITY_COLUMNS = [f"probability_class_{i}" for i in range(7)]

def write_json(path, payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf8")

def write_csv(path, frame):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding="utf8",lineterminator="\n")

class Condition:
    condition_id: str
    analysis_family: str
    start_row_zero_based: int
    time_points: int
    retained_channels_zero_based: tuple[int, ...]
    deduplicate: bool = False

def make_model(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    max_iter=5000,
                    random_state=int(seed),
                    multi_class="auto",
                ),
            ),
        ]
    )

def fit_model(model: Pipeline, features: np.ndarray, labels: np.ndarray) -> Pipeline:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        return model.fit(features, labels)

def aligned_probabilities(model: Pipeline, features: np.ndarray) -> np.ndarray:
    probability = model.predict_proba(features)
    classes = model.named_steps["model"].classes_.astype(int)
    output = np.zeros((len(features), 7), dtype=np.float64)
    output[:, classes] = probability
    return np.clip(output, 1e-12, 1.0)

def build_features(responses: np.ndarray, condition: Condition) -> np.ndarray:
    end = condition.start_row_zero_based + condition.time_points
    channels = list(condition.retained_channels_zero_based)
    if condition.start_row_zero_based < 0 or end > responses.shape[1] or not channels:
        raise RuntimeError(f"invalid feature condition: {condition}")
    return responses[:, condition.start_row_zero_based:end, :][:, :, channels].reshape(len(responses), -1).astype(np.float32)

def selected_source_mask(manifest: pd.DataFrame, deduplicate: bool) -> np.ndarray:
    if not deduplicate:
        return np.ones(len(manifest), dtype=bool)
    first_by_group = manifest.groupby("group_id", sort=False)["source_id"].transform("min")
    mask = manifest["source_id"].astype(str).eq(first_by_group.astype(str)).to_numpy()
    if int(mask.sum()) != 348 or manifest.loc[mask, "group_id"].nunique() != 348:
        raise RuntimeError("the deterministic duplicate sensitivity must retain 348 sources and groups")
    return mask

def nested_outer_fold(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    outer_fold: int,
    c1_config: dict[str, Any],
) -> dict[str, Any]:
    if set(groups[outer_train]) & set(groups[outer_test]):
        raise RuntimeError(f"outer group leakage in fold {outer_fold}")
    inner = StratifiedGroupKFold(
        n_splits=int(c1_config["inner_folds"]),
        shuffle=True,
        random_state=int(c1_config["inner_seed_base"]) + outer_fold,
    )
    inner_splits = list(inner.split(features[outer_train], labels[outer_train], groups[outer_train]))
    candidates: list[dict[str, Any]] = []
    for c_value in c1_config["candidate_c"]:
        for probability_state in c1_config["candidate_probability_states"]:
            fold_metrics: list[dict[str, float | int]] = []
            for inner_fold, (inner_train_rel, inner_validation_rel) in enumerate(inner_splits, start=1):
                inner_train = outer_train[inner_train_rel]
                inner_validation = outer_train[inner_validation_rel]
                if set(groups[inner_train]) & set(groups[inner_validation]):
                    raise RuntimeError(f"inner group leakage in outer {outer_fold}, inner {inner_fold}")
                seed = int(c1_config["outer_seed"]) + outer_fold * 100 + inner_fold
                model = fit_model(make_model(float(c_value), seed), features[inner_train], labels[inner_train])
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
                    "probability_state": str(probability_state),
                    "mean_macro_f1": float(np.mean([x["macro_f1"] for x in fold_metrics])),
                    "mean_brier_score": float(np.mean([x["brier_score"] for x in fold_metrics])),
                    "mean_ece": float(np.mean([x["expected_calibration_error"] for x in fold_metrics])),
                    "inner_folds": fold_metrics,
                }
            )
    best_macro = max(float(item["mean_macro_f1"]) for item in candidates)
    eligible = [
        item
        for item in candidates
        if float(item["mean_macro_f1"]) >= best_macro - float(c1_config["macro_f1_tie_tolerance"])
    ]
    selected = sorted(
        eligible,
        key=lambda item: (
            float(item["mean_brier_score"]),
            float(item["mean_ece"]),
            float(item["c"]),
            str(item["probability_state"]) != "raw",
        ),
    )[0]

    final_seed = int(c1_config["outer_seed"]) + outer_fold
    model = fit_model(make_model(float(selected["c"]), final_seed), features[outer_train], labels[outer_train])
    train_probability = aligned_probabilities(model, features[outer_train])
    test_probability = aligned_probabilities(model, features[outer_test])
    temperature = 1.0
    if selected["probability_state"] == "temperature_scaled":
        temperature = fit_temperature(np.log(train_probability), labels[outer_train])
        test_probability = stable_softmax(np.log(test_probability), temperature)
    grouped = aggregate_probabilities_by_group(test_probability, labels[outer_test], groups[outer_test])
    fold_metrics = classification_metrics(grouped.labels, grouped.probabilities)
    selection_rows = []
    for candidate in candidates:
        selection_rows.append(
            {
                **{key: value for key, value in candidate.items() if key != "inner_folds"},
                "selected": bool(candidate is selected),
            }
        )
    return {
        "outer_fold": outer_fold,
        "outer_train": outer_train,
        "outer_test": outer_test,
        "test_probability": test_probability,
        "outer_metrics": {
            "outer_fold": outer_fold,
            "outer_train_sources": int(len(outer_train)),
            "outer_test_sources": int(len(outer_test)),
            "selected_c": float(selected["c"]),
            "selected_probability_state": str(selected["probability_state"]),
            "fitted_temperature": float(temperature),
            **fold_metrics,
        },
        "selection_rows": selection_rows,
    }

def run_nested_condition(
    responses: np.ndarray,
    manifest: pd.DataFrame,
    condition: Condition,
    c1_config: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    all_features = build_features(responses, condition)
    mask = selected_source_mask(manifest, condition.deduplicate)
    features = all_features[mask]
    frame = manifest.loc[mask].reset_index(drop=True)
    labels = frame["class_id"].to_numpy(dtype=np.int64)
    groups = frame["group_id"].astype(str).to_numpy()
    source_ids = frame["source_id"].astype(str).to_numpy()
    outer = StratifiedGroupKFold(
        n_splits=int(c1_config["outer_folds"]),
        shuffle=True,
        random_state=int(c1_config["outer_seed"]),
    )
    splits = list(outer.split(features, labels, groups))
    tasks = []
    with ThreadPoolExecutor(max_workers=min(workers, len(splits))) as executor:
        for outer_fold, (outer_train, outer_test) in enumerate(splits, start=1):
            tasks.append(
                executor.submit(
                    nested_outer_fold,
                    features,
                    labels,
                    groups,
                    outer_train,
                    outer_test,
                    outer_fold,
                    c1_config,
                )
            )
        fold_results = sorted((future.result() for future in as_completed(tasks)), key=lambda x: int(x["outer_fold"]))

    oof_probability = np.full((len(features), 7), np.nan, dtype=np.float64)
    outer_assignment = np.full(len(features), -1, dtype=np.int64)
    outer_records: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for result in fold_results:
        test = np.asarray(result["outer_test"], dtype=np.int64)
        oof_probability[test] = np.asarray(result["test_probability"], dtype=np.float64)
        outer_assignment[test] = int(result["outer_fold"])
        outer_records.append(result["outer_metrics"])
        selection_rows.extend(result["selection_rows"])
    if np.isnan(oof_probability).any() or np.any(outer_assignment < 1):
        raise RuntimeError(f"incomplete OOF predictions for {condition.condition_id}")
    if int(pd.DataFrame({"group": groups, "fold": outer_assignment}).groupby("group")["fold"].nunique().max()) != 1:
        raise RuntimeError(f"outer group leakage for {condition.condition_id}")

    grouped = aggregate_probabilities_by_group(oof_probability, labels, groups)
    pooled_metrics = classification_metrics(grouped.labels, grouped.probabilities)
    condition_root = NESTED_ROOT / condition.condition_id
    source_predictions = pd.DataFrame(
        {
            "source_id": source_ids,
            "group_id": groups,
            "class_id": labels,
            "outer_fold": outer_assignment,
            "predicted_class_id": oof_probability.argmax(axis=1),
        }
    )
    grouped_predictions = pd.DataFrame(
        {
            "group_id": grouped.group_ids,
            "class_id": grouped.labels,
            "source_count": grouped.source_counts,
            "predicted_class_id": grouped.probabilities.argmax(axis=1),
        }
    )
    for class_id, column in enumerate(PROBABILITY_COLUMNS):
        source_predictions[column] = oof_probability[:, class_id]
        grouped_predictions[column] = grouped.probabilities[:, class_id]
    write_csv(condition_root / "nested_oof_predictions.csv", source_predictions)
    write_csv(condition_root / "nested_oof_group_predictions.csv", grouped_predictions)
    write_csv(condition_root / "inner_selection_summary.csv", pd.DataFrame(selection_rows))
    write_json(condition_root / "outer_fold_metrics.json", outer_records)
    write_json(condition_root / "pooled_metrics.json", pooled_metrics)
    metadata = {
        "schema_version": "1.0",
        "analysis_status": "post_hoc_sensitivity",
        **asdict(condition),
        "retained_channels_zero_based": list(condition.retained_channels_zero_based),
        "feature_count": int(features.shape[1]),
        "source_count": int(len(frame)),
        "group_count": int(len(grouped.group_ids)),
        "outer_group_overlap_count": 0,
        "inner_group_overlap_count": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(condition_root / "condition_metadata.json", metadata)
    summary = {
        **metadata,
        **{
            key: pooled_metrics[key]
            for key in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "expected_calibration_error",
                "multiclass_brier_score",
                "ordinal_step_mae",
            )
        },
        "selected_c_by_outer_fold": [float(item["selected_c"]) for item in outer_records],
        "selected_probability_state_by_outer_fold": [str(item["selected_probability_state"]) for item in outer_records],
    }
    print(
        f"NESTED_CONDITION_PASS id={condition.condition_id} sources={len(frame)} groups={len(grouped.group_ids)} "
        f"macro_f1={float(pooled_metrics['macro_f1']):.12f} elapsed={metadata['elapsed_seconds']:.1f}s"
    )
    return summary

def condition_definitions() -> list[Condition]:
    conditions = [
        Condition("control_t100_all8", "control", 30, 100, tuple(range(8))),
        Condition("time_t040_all8", "time_window", 30, 40, tuple(range(8))),
        Condition("time_t060_all8", "time_window", 30, 60, tuple(range(8))),
        Condition("time_t080_all8", "time_window", 30, 80, tuple(range(8))),
    ]
    for channel in range(8):
        retained = tuple(item for item in range(8) if item != channel)
        conditions.append(
            Condition(
                f"sensor_leave_out_ch{channel + 1}_t100",
                "sensor_leave_one_out",
                30,
                100,
                retained,
            )
        )
    conditions.append(
        Condition(
            "deduplicated_t100_all8",
            "duplicate_group_sensitivity",
            30,
            100,
            tuple(range(8)),
            deduplicate=True,
        )
    )
    return conditions

def main():
    global NESTED_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=[c.condition_id for c in condition_definitions()])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=NESTED_ROOT)
    args=parser.parse_args()
    if args.workers < 1: parser.error("workers must be positive")
    NESTED_ROOT=args.output.resolve()
    config=json.loads((ROOT/"config/benchmark.json").read_text(encoding="utf8"))
    responses=np.load(ROOT/"data/source_responses.npz",allow_pickle=False)["responses"]
    manifest=pd.read_csv(ROOT/"data/source_manifest.csv")
    selected=[c for c in condition_definitions() if args.condition is None or c.condition_id==args.condition]
    rows=[run_nested_condition(responses,manifest,c,config,args.workers) for c in selected]
    write_csv(NESTED_ROOT/"summary.csv",pd.DataFrame(rows))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
