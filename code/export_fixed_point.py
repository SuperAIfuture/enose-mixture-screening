"""Export the five C1 fold models as multi-bit linear memories."""
from __future__ import annotations
import argparse,hashlib,json,math,os
from pathlib import Path
from typing import Any,Iterable
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name,"1")
import numpy as np
import pandas as pd
from run_benchmark import make_model
ROOT=Path(__file__).resolve().parents[1]
PACKAGE_ROOT=ROOT
FOLD_ROOT=ROOT/"outputs/c1_fixed_point/folds"

def fit_model(model,x,y):
    return model.fit(x,y)

def aligned_decision_scores(model,x):
    return model.decision_function(x)

def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()

def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf8")

def write_csv(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding="utf8",lineterminator="\n")

def write_text(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value,encoding="utf8",newline="\n")

def signed_hex(value: int, width_bits: int) -> str:
    minimum = -(1 << (width_bits - 1))
    maximum = (1 << (width_bits - 1)) - 1
    if value < minimum or value > maximum:
        raise OverflowError(f"{value} does not fit signed {width_bits}-bit representation")
    encoded = value if value >= 0 else (1 << width_bits) + value
    digits = (width_bits + 3) // 4
    return f"{encoded:0{digits}X}"

def unsigned_hex(value: int, width_bits: int) -> str:
    if value < 0 or value >= (1 << width_bits):
        raise OverflowError(f"{value} does not fit unsigned {width_bits}-bit representation")
    digits = (width_bits + 3) // 4
    return f"{value:0{digits}X}"

def write_mem(path: Path, values: Iterable[int], width_bits: int, signed: bool) -> None:
    encoder = signed_hex if signed else unsigned_hex
    write_text(path, "".join(f"{encoder(int(value), width_bits)}\n" for value in values))

def build_fixed_point_fold(
    fold: int,
    features: np.ndarray,
    manifest: pd.DataFrame,
    baseline: pd.DataFrame,
    outer_metric: dict[str, Any],
    c1_config: dict[str, Any],
    fixed_config: dict[str, Any],
) -> dict[str, Any]:
    train = np.flatnonzero(baseline["outer_fold"].to_numpy(dtype=np.int64) != fold)
    test = np.flatnonzero(baseline["outer_fold"].to_numpy(dtype=np.int64) == fold)
    groups = manifest["group_id"].astype(str).to_numpy()
    if set(groups[train]) & set(groups[test]):
        raise RuntimeError(f"fixed-point fold {fold} crosses immutable groups")
    labels = manifest["class_id"].to_numpy(dtype=np.int64)
    selected_c = float(outer_metric["selected_c"])
    model = fit_model(
        make_model(selected_c, int(c1_config["outer_seed"]) + fold),
        features[train],
        labels[train],
    )
    scaler = model.named_steps["scale"]
    classifier = model.named_steps["model"]
    effective_weight = classifier.coef_.astype(np.float64) / scaler.scale_.astype(np.float64)[None, :]
    effective_bias = classifier.intercept_.astype(np.float64) - np.sum(
        classifier.coef_.astype(np.float64)
        * scaler.mean_.astype(np.float64)[None, :]
        / scaler.scale_.astype(np.float64)[None, :],
        axis=1,
    )
    float_score = aligned_decision_scores(model, features[test])
    float_prediction = float_score.argmax(axis=1).astype(np.int64)
    baseline_prediction = baseline.iloc[test]["predicted_class_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(float_prediction, baseline_prediction):
        raise RuntimeError(f"refitted floating fold {fold} differs from the frozen C1 decision")

    input_width = int(fixed_config["input_width_bits"])
    input_scale = int(fixed_config["input_scale"])
    weight_width = int(fixed_config["weight_width_bits"])
    accumulator_width = int(fixed_config["accumulator_width_bits"])
    maximum_fractional_bits = int(fixed_config["maximum_weight_fractional_bits"])
    maximum_weight = float(np.max(np.abs(effective_weight)))
    maximum_integer = (1 << (weight_width - 1)) - 1
    allowed_scale = maximum_integer / maximum_weight
    fractional_bits = min(maximum_fractional_bits, int(math.floor(math.log2(allowed_scale))))
    if fractional_bits < 0:
        raise RuntimeError(f"fold {fold} effective weights do not fit the signed weight width")
    weight_scale = 1 << fractional_bits
    quantized_weight = np.rint(effective_weight * weight_scale).astype(np.int64)
    quantized_bias = np.rint(effective_bias * input_scale * weight_scale).astype(np.int64)
    if np.max(np.abs(quantized_weight)) > maximum_integer:
        raise RuntimeError(f"fold {fold} quantized weights overflow")
    accumulator_limit = (1 << (accumulator_width - 1)) - 1
    if np.max(np.abs(quantized_bias)) > accumulator_limit:
        raise RuntimeError(f"fold {fold} quantized biases overflow")
    test_features = features[test]
    if float(test_features.min()) < 0.0 or float(test_features.max()) > 1.0:
        raise RuntimeError("the unsigned input mapping requires normalized features in [0, 1]")
    quantized_input = np.rint(test_features * input_scale).astype(np.int64)
    fixed_score = quantized_input @ quantized_weight.T + quantized_bias[None, :]
    if np.max(np.abs(fixed_score)) > accumulator_limit:
        raise RuntimeError(f"fold {fold} accumulated score exceeds signed 64-bit range")
    fixed_prediction = fixed_score.argmax(axis=1).astype(np.int64)
    denominator = float(input_scale * weight_scale)
    reconstructed_score = fixed_score.astype(np.float64) / denominator
    fixed_float_agreement = int(np.sum(fixed_prediction == float_prediction))
    fold_root = FOLD_ROOT / f"fold_{fold:02d}"
    fold_root.mkdir(parents=True, exist_ok=True)
    write_mem(fold_root / "weights.mem", quantized_weight.reshape(-1), weight_width, signed=True)
    write_mem(fold_root / "biases.mem", quantized_bias, accumulator_width, signed=True)
    write_mem(fold_root / "inputs.mem", quantized_input.reshape(-1), input_width, signed=False)
    write_mem(fold_root / "expected_fixed_classes.mem", fixed_prediction, 3, signed=False)
    write_mem(fold_root / "expected_float_classes.mem", float_prediction, 3, signed=False)
    source_frame = baseline.iloc[test][["source_id", "group_id", "class_id", "outer_fold"]].reset_index(drop=True)
    source_frame["floating_class_id"] = float_prediction
    source_frame["fixed_point_class_id"] = fixed_prediction
    source_frame["fixed_matches_floating"] = fixed_prediction == float_prediction
    write_csv(fold_root / "source_manifest.csv", source_frame)
    result = {
        "fold": fold,
        "training_source_count": int(len(train)),
        "test_source_count": int(len(test)),
        "test_group_count": int(manifest.iloc[test]["group_id"].nunique()),
        "selected_c": selected_c,
        "frozen_probability_state": str(outer_metric["selected_probability_state"]),
        "mapping_probability_state": "raw_decision_score",
        "input_width_bits": input_width,
        "input_scale": input_scale,
        "weight_width_bits": weight_width,
        "weight_fractional_bits": fractional_bits,
        "weight_scale": weight_scale,
        "accumulator_width_bits": accumulator_width,
        "maximum_absolute_effective_weight": maximum_weight,
        "maximum_absolute_quantized_weight": int(np.max(np.abs(quantized_weight))),
        "maximum_absolute_quantized_bias": int(np.max(np.abs(quantized_bias))),
        "maximum_absolute_accumulated_score": int(np.max(np.abs(fixed_score))),
        "maximum_input_quantization_error": float(np.max(np.abs(quantized_input / input_scale - test_features))),
        "maximum_reconstructed_score_error": float(np.max(np.abs(reconstructed_score - float_score))),
        "floating_vs_frozen_c1_decision_agreement": int(np.sum(float_prediction == baseline_prediction)),
        "fixed_vs_floating_decision_agreement": fixed_float_agreement,
        "fixed_vs_floating_decision_agreement_fraction": float(fixed_float_agreement / len(test)),
        "class_tie_rule": "lowest class index wins because updates require strictly greater score",
        "memory_files": {
            name: {
                "path": (fold_root / name).relative_to(PACKAGE_ROOT).as_posix(),
                "sha256": sha256_file(fold_root / name),
            }
            for name in (
                "weights.mem",
                "biases.mem",
                "inputs.mem",
                "expected_fixed_classes.mem",
                "expected_float_classes.mem",
                "source_manifest.csv",
            )
        },
    }
    write_json(fold_root / "fold_quantization.json", result)
    print(
        f"FIXED_POINT_FOLD_PASS fold={fold} fixed_float={fixed_float_agreement}/{len(test)} "
        f"fractional_bits={fractional_bits} max_acc={result['maximum_absolute_accumulated_score']}"
    )
    return result

def main():
    global FOLD_ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/c1_fixed_point")
    args=parser.parse_args()
    FOLD_ROOT=args.output.resolve()/"folds"
    c=json.loads((ROOT/"config/benchmark.json").read_text(encoding="utf8"))
    fixed=json.loads((ROOT/"config/analyses.json").read_text(encoding="utf8"))["fixed_point"]
    x=np.load(ROOT/"data/source_responses.npz",allow_pickle=False)["responses"][:,30:130,:].reshape(350,800).astype(np.float32)
    m=pd.read_csv(ROOT/"data/source_manifest.csv")
    b=pd.read_csv(ROOT/"results/benchmark/nested_oof_predictions.csv")
    folds=json.loads((ROOT/"results/benchmark/outer_fold_metrics.json").read_text(encoding="utf8"))
    rows=[build_fixed_point_fold(int(f["outer_fold"]),x,m,b,f,c,fixed) for f in folds]
    total=sum(r["fixed_vs_floating_decision_agreement"] for r in rows)
    write_json(args.output.resolve()/"quantization.json",rows)
    assert total==350
    print("FIXED_POINT_EXPORT_PASS folds=5 decisions=350/350")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
