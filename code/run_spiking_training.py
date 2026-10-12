#!/usr/bin/env python3
"""Train quantized spiking candidates with group-preserving partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader, TensorDataset

from metrics_utils import (
    ORDINAL_POSITION_BY_CLASS,
    aggregate_probabilities_by_group,
    classification_metrics,
    fit_temperature,
    stable_softmax,
    stratified_group_bootstrap_macro_f1,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "spiking_training.json"
RESULTS = ROOT / "outputs" / "spiking_training"
RUN_RESULTS = RESULTS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def artifact_reference(path: Path) -> str:
    """Store artifact paths relative to this repository root for portable result bundles."""
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def resolve_artifact_reference(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def load_inputs() -> tuple[np.ndarray, pd.DataFrame]:
    with np.load(ROOT / "data" / "source_responses.npz", allow_pickle=False) as archive:
        responses = archive["responses"].astype(np.float32)
    manifest = pd.read_csv(ROOT / "data" / "source_manifest.csv")
    if responses.shape != (350, 150, 8) or len(manifest) != 350:
        raise RuntimeError(f"unexpected packaged data: responses={responses.shape}, manifest={len(manifest)}")
    inputs = responses[:, 30:130, :].reshape(350, 800).astype(np.float32)
    if not np.isfinite(inputs).all():
        raise RuntimeError("non-finite input detected")
    return inputs, manifest


def load_split(split_id: str, manifest: pd.DataFrame) -> pd.DataFrame:
    split = pd.read_csv(ROOT / "data" / "spiking_splits" / f"{split_id}.csv")
    merged = manifest[["source_id", "class_id", "class_label", "group_id", "ordinal_index"]].merge(
        split[["source_id", "partition"]], on="source_id", how="left", validate="one_to_one"
    )
    if merged["partition"].isna().any():
        raise RuntimeError(f"missing partition in {split_id}")
    overlap = merged.groupby("group_id")["partition"].nunique()
    if int((overlap > 1).sum()) != 0:
        raise RuntimeError(f"group overlap detected in {split_id}")
    return merged


def build_training_pairs(
    responses: np.ndarray,
    metadata: pd.DataFrame,
    split_id: str,
    seed: int,
    views_per_source: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    training_indices = np.flatnonzero(metadata["partition"].to_numpy() == "train")
    labels = metadata["class_id"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    original_views: list[np.ndarray] = []
    augmented_views: list[np.ndarray] = []
    pair_labels: list[int] = []
    lineage: list[dict[str, object]] = []
    for source_index in training_indices:
        source = responses[source_index]
        class_id = int(labels[source_index])
        donor_pool = training_indices[labels[training_indices] == class_id]
        donor_pool = donor_pool[donor_pool != source_index]
        if donor_pool.size == 0:
            raise RuntimeError(f"empty donor pool for source {source_index}")
        for view_index in range(views_per_source):
            donor_index = int(rng.choice(donor_pool))
            channels = np.sort(rng.choice(8, size=4, replace=False))
            augmented = source.copy()
            augmented[:, channels] = responses[donor_index][:, channels]
            noise_scale = source.std(axis=0) * 0.02
            augmented = np.clip(augmented + rng.normal(0.0, noise_scale, size=source.shape), 0.0, 1.0).astype(np.float32)
            original_views.append(source[30:130].reshape(-1))
            augmented_views.append(augmented[30:130].reshape(-1))
            pair_labels.append(class_id)
            lineage.append(
                {
                    "split_id": split_id,
                    "augmentation_seed": seed,
                    "parent_source_id": metadata.iloc[source_index]["source_id"],
                    "parent_group_id": metadata.iloc[source_index]["group_id"],
                    "donor_source_id": metadata.iloc[donor_index]["source_id"],
                    "donor_group_id": metadata.iloc[donor_index]["group_id"],
                    "class_id": class_id,
                    "view_index": view_index + 1,
                    "swapped_channels": ";".join(str(int(value)) for value in channels),
                    "partition": "train",
                }
            )
    return (
        np.stack(original_views).astype(np.float32),
        np.stack(augmented_views).astype(np.float32),
        np.asarray(pair_labels, dtype=np.int64),
        lineage,
    )


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return (value > 0).to(value.dtype)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        (value,) = ctx.saved_tensors
        surrogate = 1.0 / torch.square(1.0 + 5.0 * value.abs())
        return (gradient * surrogate,)


def quantize_membrane_ste(value: torch.Tensor, step: float = 1.0 / 128.0) -> torch.Tensor:
    quantized = torch.round(value / step) * step
    return value + (quantized - value).detach()


class QuantizedSparseLinear(nn.Module):
    def __init__(self, input_features: int, output_features: int, fixed_mask: np.ndarray | None = None) -> None:
        super().__init__()
        if input_features % 4 != 0:
            raise ValueError("2:4 sparsity requires input_features divisible by four")
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5.0))
        if fixed_mask is None:
            grouped_weight = self.weight.detach().abs().reshape(output_features, input_features // 4, 4)
            keep = torch.topk(grouped_weight, k=2, dim=2, largest=True, sorted=False).indices
            grouped_mask = torch.zeros_like(grouped_weight)
            grouped_mask.scatter_(2, keep, 1.0)
            fixed_mask = grouped_mask.reshape(output_features, input_features).cpu().numpy()
        fixed_mask_array = np.asarray(fixed_mask, dtype=np.float32)
        if fixed_mask_array.shape != (output_features, input_features):
            raise ValueError("fixed 2:4 mask shape does not match the linear layer")
        if not np.all(fixed_mask_array.reshape(output_features, input_features // 4, 4).sum(axis=2) == 2):
            raise ValueError("fixed 2:4 mask must retain exactly two weights in every group of four")
        self.register_buffer("fixed_mask", torch.from_numpy(fixed_mask_array))

    def sparse_mask(self) -> torch.Tensor:
        return self.fixed_mask

    def fake_quantized_weight(self) -> torch.Tensor:
        sparse = self.weight * self.sparse_mask()
        scale = torch.clamp(sparse.detach().abs().max() / 127.0, min=1e-8)
        integer = torch.clamp(torch.round(sparse / scale), -127, 127)
        dequantized = integer * scale
        return sparse + (dequantized - sparse).detach()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return functional.linear(value, self.fake_quantized_weight())

    def export_quantized(self) -> tuple[np.ndarray, float, np.ndarray]:
        with torch.no_grad():
            mask = self.sparse_mask()
            sparse = self.weight * mask
            scale = float(torch.clamp(sparse.abs().max() / 127.0, min=1e-8).cpu())
            integer = torch.clamp(torch.round(sparse / scale), -127, 127).to(torch.int8).cpu().numpy()
        return integer, scale, mask.to(torch.uint8).cpu().numpy()


class TLOCQSNN(nn.Module):
    def __init__(self, hidden_width: int, time_steps: int, probability_scale: float) -> None:
        super().__init__()
        self.hidden_width = hidden_width
        self.time_steps = time_steps
        self.probability_scale = probability_scale
        if hidden_width == 0:
            self.output_layer = QuantizedSparseLinear(800, 7)
            self.hidden_layer = None
        else:
            self.hidden_layer = QuantizedSparseLinear(800, hidden_width)
            self.output_layer = QuantizedSparseLinear(hidden_width, 7)

    def encode(self, value: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        probability = torch.clamp(value * self.probability_scale, 0.0, 1.0)
        random_value = torch.rand(
            (self.time_steps, value.shape[0], value.shape[1]),
            device=value.device,
            generator=generator,
        )
        return (random_value < probability.unsqueeze(0)).to(value.dtype)

    @staticmethod
    def lif_step(current: torch.Tensor, membrane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        membrane = quantize_membrane_ste(0.5 * membrane + current)
        spike = SurrogateSpike.apply(membrane - 1.0)
        membrane = membrane * (1.0 - spike)
        return spike, membrane

    def forward_spikes(self, spikes: torch.Tensor) -> torch.Tensor:
        time_steps, batch = spikes.shape[:2]
        output_membrane = torch.zeros(batch, 7, device=spikes.device)
        output_count = torch.zeros(batch, 7, device=spikes.device)
        hidden_membrane = (
            torch.zeros(batch, self.hidden_width, device=spikes.device) if self.hidden_layer is not None else None
        )
        flattened = spikes.reshape(time_steps * batch, spikes.shape[2])
        if self.hidden_layer is not None:
            hidden_weight = self.hidden_layer.fake_quantized_weight()
            hidden_current = functional.linear(flattened, hidden_weight).reshape(time_steps, batch, self.hidden_width)
            hidden_spikes: list[torch.Tensor] = []
            for frame_current in hidden_current:
                hidden_spike, hidden_membrane = self.lif_step(frame_current, hidden_membrane)
                hidden_spikes.append(hidden_spike)
            hidden_spike_tensor = torch.stack(hidden_spikes, dim=0)
            output_weight = self.output_layer.fake_quantized_weight()
            output_current = functional.linear(
                hidden_spike_tensor.reshape(time_steps * batch, self.hidden_width), output_weight
            ).reshape(time_steps, batch, 7)
        else:
            output_weight = self.output_layer.fake_quantized_weight()
            output_current = functional.linear(flattened, output_weight).reshape(time_steps, batch, 7)
        for frame_current in output_current:
            output_spike, output_membrane = self.lif_step(frame_current, output_membrane)
            output_count = output_count + output_spike
        return output_count / float(time_steps)

    def forward(self, value: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.forward_spikes(self.encode(value, generator))


def ordinal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positions = torch.as_tensor(ORDINAL_POSITION_BY_CLASS, device=logits.device, dtype=logits.dtype)
    truth_position = positions[labels].unsqueeze(1)
    distance = torch.abs(positions.unsqueeze(0) - truth_position) / 6.0
    return torch.mean(torch.sum(torch.softmax(logits, dim=1) * distance, dim=1))


def consistency_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_probability = torch.softmax(first, dim=1)
    second_probability = torch.softmax(second, dim=1)
    midpoint = torch.clamp((first_probability + second_probability) / 2.0, min=1e-8)
    first_log = torch.log(torch.clamp(first_probability, min=1e-8))
    second_log = torch.log(torch.clamp(second_probability, min=1e-8))
    midpoint_log = torch.log(midpoint)
    return 0.5 * (
        functional.kl_div(midpoint_log, first_probability, reduction="batchmean", log_target=False)
        + functional.kl_div(midpoint_log, second_probability, reduction="batchmean", log_target=False)
    )


@torch.no_grad()
def evaluate_logits(
    model: TLOCQSNN,
    inputs: np.ndarray,
    *,
    device: torch.device,
    repeats: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(inputs).to(device)
    repeat_logits: list[np.ndarray] = []
    for repeat in range(repeats):
        generator = torch.Generator(device=device.type)
        generator.manual_seed(seed + repeat * 100003)
        batches: list[np.ndarray] = []
        for start in range(0, tensor.shape[0], batch_size):
            batches.append(model(tensor[start : start + batch_size], generator).cpu().numpy())
        repeat_logits.append(np.concatenate(batches))
    return np.mean(np.stack(repeat_logits), axis=0)


def grouped_macro_f1_from_logits(logits: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    grouped = aggregate_probabilities_by_group(stable_softmax(logits), labels, groups)
    return float(classification_metrics(grouped.labels, grouped.probabilities)["macro_f1"])


def train_snn_candidate(
    *,
    hidden_width: int,
    split_id: str,
    training_seed: int,
    responses: np.ndarray,
    inputs: np.ndarray,
    metadata: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    maximum_epochs: int,
) -> dict[str, object]:
    set_seed(training_seed + hidden_width * 101)
    training = config["training"]
    original, augmented, labels, lineage = build_training_pairs(
        responses,
        metadata,
        split_id,
        seed=training_seed,
        views_per_source=int(training["augmentation_views_per_source"]),
    )
    lineage_dir = RUN_RESULTS / "augmentation_manifests"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = lineage_dir / f"{split_id}_seed_{training_seed}.csv"
    if not lineage_path.exists():
        pd.DataFrame(lineage).to_csv(lineage_path, index=False, encoding="utf-8")

    dataset = TensorDataset(
        torch.from_numpy(original),
        torch.from_numpy(augmented),
        torch.from_numpy(labels),
    )
    loader_generator = torch.Generator().manual_seed(training_seed)
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )
    model = TLOCQSNN(
        hidden_width,
        time_steps=int(config["data"]["time_steps"]),
        probability_scale=float(training["spike_probability_scale"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    validation_index = np.flatnonzero(metadata["partition"].to_numpy() == "validation")
    validation_inputs = inputs[validation_index]
    validation_labels = metadata.iloc[validation_index]["class_id"].to_numpy(dtype=np.int64)
    validation_groups = metadata.iloc[validation_index]["group_id"].astype(str).to_numpy()
    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    patience = int(training["early_stopping_patience"])
    no_improvement = 0
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for original_batch, augmented_batch, label_batch in loader:
            original_batch = original_batch.to(device)
            augmented_batch = augmented_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            # Encode both parent views in one batch. The two halves still receive
            # independent Bernoulli draws while sharing the vectorized forward path.
            paired_logits = model(torch.cat((original_batch, augmented_batch), dim=0))
            original_logits, augmented_logits = torch.split(paired_logits, original_batch.shape[0], dim=0)
            cross_entropy = 0.5 * (
                functional.cross_entropy(original_logits, label_batch)
                + functional.cross_entropy(augmented_logits, label_batch)
            )
            ordinal = 0.5 * (
                ordinal_loss(original_logits, label_batch) + ordinal_loss(augmented_logits, label_batch)
            )
            consistency = consistency_loss(original_logits, augmented_logits)
            loss = (
                float(training["cross_entropy_weight"]) * cross_entropy
                + float(training["ordinal_loss_weight"]) * ordinal
                + float(training["parent_view_consistency_weight"]) * consistency
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * label_batch.shape[0]
            examples += int(label_batch.shape[0])
        validation_logits = evaluate_logits(
            model,
            validation_inputs,
            device=device,
        repeats=int(training["evaluation_encoding_repeats"]),
        seed=training_seed + 700000 + epoch,
        batch_size=max(int(training["batch_size"]), len(validation_inputs)),
        )
        validation_score = grouped_macro_f1_from_logits(validation_logits, validation_labels, validation_groups)
        history.append({"epoch": epoch, "training_loss": total_loss / examples, "validation_grouped_macro_f1": validation_score})
        if validation_score > best_score + 1e-8:
            best_score = validation_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    final_validation_logits = evaluate_logits(
        model,
        validation_inputs,
        device=device,
        repeats=int(training["evaluation_encoding_repeats"]),
        seed=training_seed + 800001,
        batch_size=max(int(training["batch_size"]), len(validation_inputs)),
    )
    final_validation_grouped = aggregate_probabilities_by_group(
        stable_softmax(final_validation_logits), validation_labels, validation_groups
    )
    final_validation_metrics = classification_metrics(
        final_validation_grouped.labels, final_validation_grouped.probabilities
    )
    mask_values = [
        module.sparse_mask().detach().cpu().numpy()
        for module in model.modules()
        if isinstance(module, QuantizedSparseLinear)
    ]
    verified_structured_sparsity = float(
        1.0 - np.concatenate([value.reshape(-1) for value in mask_values]).mean()
    )
    checkpoint_dir = RUN_RESULTS / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{split_id}_width_{hidden_width}_seed_{training_seed}.pt"
    torch.save(
        {
            "model_state": best_state,
            "hidden_width": hidden_width,
            "time_steps": int(config["data"]["time_steps"]),
            "probability_scale": float(training["spike_probability_scale"]),
            "split_id": split_id,
            "training_seed": training_seed,
            "mask_formation_seed": training_seed + hidden_width * 101,
            "best_epoch": best_epoch,
            "best_validation_grouped_macro_f1": best_score,
        },
        checkpoint_path,
    )
    history_path = RUN_RESULTS / "training_history" / f"{split_id}_width_{hidden_width}_seed_{training_seed}.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8")
    return {
        "split_id": split_id,
        "hidden_width": hidden_width,
        "training_seed": training_seed,
        "best_epoch": best_epoch,
        "validation_grouped_macro_f1": float(final_validation_metrics["macro_f1"]),
        "validation_expected_calibration_error": float(final_validation_metrics["expected_calibration_error"]),
        "verified_structured_sparsity": verified_structured_sparsity,
        "early_stopping_best_validation_grouped_macro_f1": best_score,
        "checkpoint": artifact_reference(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history": artifact_reference(history_path),
        "augmentation_manifest": artifact_reference(lineage_path),
    }


def load_checkpoint(path: Path, device: torch.device) -> tuple[TLOCQSNN, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TLOCQSNN(
        int(checkpoint["hidden_width"]),
        int(checkpoint["time_steps"]),
        float(checkpoint["probability_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def evaluate_selected_snn(
    selection: dict[str, Any],
    validation_rows: pd.DataFrame,
    inputs: np.ndarray,
    manifests: dict[str, pd.DataFrame],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    training = config["training"]
    width = int(selection["selected_hidden_width"])
    split_metrics: list[dict[str, object]] = []
    source_prediction_frames: list[pd.DataFrame] = []
    group_prediction_frames: list[pd.DataFrame] = []
    available_splits = [str(value) for value in validation_rows["split_id"].drop_duplicates().tolist()]
    available_seeds = [int(value) for value in validation_rows["training_seed"].drop_duplicates().tolist()]
    for split_id, metadata in manifests.items():
        validation_index = np.flatnonzero(metadata["partition"].to_numpy() == "validation")
        test_index = np.flatnonzero(metadata["partition"].to_numpy() == "test")
        calibrated_probabilities: list[np.ndarray] = []
        temperatures: list[float] = []
        for training_seed in available_seeds:
            row = validation_rows[
                (validation_rows["split_id"] == split_id)
                & (validation_rows["hidden_width"] == width)
                & (validation_rows["training_seed"] == int(training_seed))
            ].iloc[0]
            model, _ = load_checkpoint(resolve_artifact_reference(row["checkpoint"]), device)
            validation_logits = evaluate_logits(
                model,
                inputs[validation_index],
                device=device,
                repeats=int(training["evaluation_encoding_repeats"]),
                seed=int(training_seed) + 900001,
                batch_size=max(int(training["batch_size"]), len(validation_index)),
            )
            validation_labels = metadata.iloc[validation_index]["class_id"].to_numpy(dtype=np.int64)
            temperature = fit_temperature(validation_logits, validation_labels)
            test_logits = evaluate_logits(
                model,
                inputs[test_index],
                device=device,
                repeats=int(training["evaluation_encoding_repeats"]),
                seed=int(training_seed) + 1900001,
                batch_size=max(int(training["batch_size"]), len(test_index)),
            )
            calibrated_probabilities.append(stable_softmax(test_logits, temperature))
            temperatures.append(temperature)
        probabilities = np.mean(np.stack(calibrated_probabilities), axis=0)
        test_metadata = metadata.iloc[test_index].reset_index(drop=True)
        grouped = aggregate_probabilities_by_group(
            probabilities,
            test_metadata["class_id"].to_numpy(dtype=np.int64),
            test_metadata["group_id"].astype(str).to_numpy(),
        )
        metrics = classification_metrics(grouped.labels, grouped.probabilities)
        split_metrics.append(
            {
                "model": "TLOC-QSNN",
                "split_id": split_id,
                "hidden_width": width,
                "temperature_by_seed": temperatures,
                **metrics,
            }
        )
        source_frame = test_metadata[["source_id", "group_id", "class_id", "class_label", "ordinal_index"]].copy()
        source_frame.insert(0, "split_id", split_id)
        for class_id in range(7):
            source_frame[f"probability_class_{class_id}"] = probabilities[:, class_id]
        source_frame["predicted_class_id"] = probabilities.argmax(axis=1)
        source_prediction_frames.append(source_frame)
        group_frame = pd.DataFrame(
            {
                "split_id": split_id,
                "group_id": grouped.group_ids,
                "class_id": grouped.labels,
                "source_count": grouped.source_counts,
                "predicted_class_id": grouped.probabilities.argmax(axis=1),
            }
        )
        for class_id in range(7):
            group_frame[f"probability_class_{class_id}"] = grouped.probabilities[:, class_id]
        group_prediction_frames.append(group_frame)
    return split_metrics, pd.concat(source_prediction_frames, ignore_index=True), pd.concat(group_prediction_frames, ignore_index=True)


def conventional_models(seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=3000, random_state=seed))]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, random_state=seed),
        "sklearn_mlp": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        alpha=1e-4,
                        early_stopping=True,
                        validation_fraction=0.15,
                        max_iter=500,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def run_conventional_baselines(
    responses: np.ndarray,
    inputs: np.ndarray,
    manifests: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    output_metrics: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    baseline_dir = RUN_RESULTS / "baseline_models"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_seed = int(config["model_selection"]["training_seeds"][0])
    for split_id, metadata in manifests.items():
        original, augmented, pair_labels, _ = build_training_pairs(
            responses,
            metadata,
            split_id,
            seed=baseline_seed,
            views_per_source=int(config["training"]["augmentation_views_per_source"]),
        )
        train_input = np.concatenate([original, augmented], axis=0)
        train_labels = np.concatenate([pair_labels, pair_labels], axis=0)
        validation_index = np.flatnonzero(metadata["partition"].to_numpy() == "validation")
        test_index = np.flatnonzero(metadata["partition"].to_numpy() == "test")
        validation_labels = metadata.iloc[validation_index]["class_id"].to_numpy(dtype=np.int64)
        test_metadata = metadata.iloc[test_index].reset_index(drop=True)
        for model_name, model in conventional_models(baseline_seed).items():
            model.fit(train_input, train_labels)
            validation_probability = np.clip(model.predict_proba(inputs[validation_index]), 1e-12, 1.0)
            temperature = fit_temperature(np.log(validation_probability), validation_labels)
            test_probability = np.clip(model.predict_proba(inputs[test_index]), 1e-12, 1.0)
            test_probability = stable_softmax(np.log(test_probability), temperature)
            grouped = aggregate_probabilities_by_group(
                test_probability,
                test_metadata["class_id"].to_numpy(dtype=np.int64),
                test_metadata["group_id"].astype(str).to_numpy(),
            )
            metrics = classification_metrics(grouped.labels, grouped.probabilities)
            model_path = baseline_dir / f"{split_id}_{model_name}.joblib"
            joblib.dump(model, model_path)
            output_metrics.append(
                {
                    "model": model_name,
                    "split_id": split_id,
                    "temperature": temperature,
                    "model_path": artifact_reference(model_path),
                    "model_sha256": sha256_file(model_path),
                    **metrics,
                }
            )
            frame = pd.DataFrame(
                {
                    "model": model_name,
                    "split_id": split_id,
                    "group_id": grouped.group_ids,
                    "class_id": grouped.labels,
                    "source_count": grouped.source_counts,
                    "predicted_class_id": grouped.probabilities.argmax(axis=1),
                }
            )
            for class_id in range(7):
                frame[f"probability_class_{class_id}"] = grouped.probabilities[:, class_id]
            prediction_frames.append(frame)
    return output_metrics, pd.concat(prediction_frames, ignore_index=True)


def select_architecture(validation_rows: pd.DataFrame, config: dict[str, Any]) -> dict[str, object]:
    summary = (
        validation_rows.groupby("hidden_width")
        .agg(
            validation_macro_f1_mean=("validation_grouped_macro_f1", "mean"),
            validation_macro_f1_std=("validation_grouped_macro_f1", "std"),
            validation_macro_f1_min=("validation_grouped_macro_f1", "min"),
            validation_macro_f1_max=("validation_grouped_macro_f1", "max"),
            validation_ece_mean=("validation_expected_calibration_error", "mean"),
            verified_structured_sparsity_mean=("verified_structured_sparsity", "mean"),
            count=("validation_grouped_macro_f1", "count"),
        )
        .reset_index()
        .sort_values("hidden_width")
    )
    best_mean = float(summary["validation_macro_f1_mean"].max())
    tolerance = float(config["model_selection"]["tie_tolerance_macro_f1"])
    eligible = summary[summary["validation_macro_f1_mean"] >= best_mean - tolerance].copy()
    min_ece = float(eligible["validation_ece_mean"].min())
    ece_eligible = eligible[eligible["validation_ece_mean"] <= min_ece + 1e-12].copy()
    selected_width = int(ece_eligible.sort_values("hidden_width").iloc[0]["hidden_width"])
    return {
        "selected_hidden_width": selected_width,
        "selection_metric": config["model_selection"]["selection_metric"],
        "best_mean_validation_macro_f1": best_mean,
        "best_mean_validation_ece_among_eligible": min_ece,
        "tie_tolerance_macro_f1": tolerance,
        "eligible_widths_within_tolerance": eligible["hidden_width"].astype(int).tolist(),
        "ece_eligible_widths": ece_eligible["hidden_width"].astype(int).tolist(),
        "selection_used_test_data": False,
        "summary": summary.to_dict(orient="records"),
    }


def export_c2_pack(
    selection: dict[str, Any],
    validation_rows: pd.DataFrame,
    inputs: np.ndarray,
    manifests: dict[str, pd.DataFrame],
    config: dict[str, Any],
    device: torch.device,
    *,
    smoke_run: bool = False,
) -> dict[str, object]:
    split_id = config["splitting"]["primary_split_id"]
    training_seed = int(config["final_prediction"]["c2_primary_training_seed"])
    width = int(selection["selected_hidden_width"])
    row = validation_rows[
        (validation_rows["split_id"] == split_id)
        & (validation_rows["hidden_width"] == width)
        & (validation_rows["training_seed"] == training_seed)
    ].iloc[0]
    model, checkpoint = load_checkpoint(resolve_artifact_reference(row["checkpoint"]), device)
    export_dir = RUN_RESULTS / ("c2_smoke_pack" if smoke_run else "c2_frozen_pack")
    export_dir.mkdir(parents=True, exist_ok=True)
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    masks: dict[str, np.ndarray] = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedSparseLinear):
            integer, scale, mask = module.export_quantized()
            safe_name = name.replace(".", "_")
            quantized[f"{safe_name}_weight_int8"] = integer
            masks[f"{safe_name}_mask_2of4"] = mask
            scales[safe_name] = scale
    weight_path = export_dir / "selected_model_int8.npz"
    np.savez_compressed(weight_path, **quantized, **masks)

    metadata = manifests[split_id]
    test_index = np.flatnonzero(metadata["partition"].to_numpy() == "test")
    frozen_inputs = inputs[test_index]
    frozen_metadata = metadata.iloc[test_index].reset_index(drop=True)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(training_seed + 2900001)
    input_tensor = torch.from_numpy(frozen_inputs).to(device)
    with torch.no_grad():
        spikes = model.encode(input_tensor, generator)
        logits = model.forward_spikes(spikes)
    frames = spikes.permute(1, 0, 2).to(torch.uint8).cpu().numpy()
    logits_numpy = logits.cpu().numpy()
    vector_path = export_dir / "split_01_frozen_vectors.npz"
    np.savez_compressed(
        vector_path,
        inputs=frozen_inputs,
        spike_frames=frames,
        labels=frozen_metadata["class_id"].to_numpy(dtype=np.int16),
        source_ids=frozen_metadata["source_id"].astype(str).to_numpy(),
        group_ids=frozen_metadata["group_id"].astype(str).to_numpy(),
        software_logits=logits_numpy,
        software_predictions=logits_numpy.argmax(axis=1).astype(np.int16),
    )
    vector_manifest = frozen_metadata[["source_id", "group_id", "class_id", "class_label", "ordinal_index"]].copy()
    vector_manifest["software_prediction_class_id"] = logits_numpy.argmax(axis=1)
    vector_manifest.to_csv(export_dir / "split_01_frozen_vector_manifest.csv", index=False, encoding="utf-8")
    specification = {
        "pack_status": "pipeline_smoke_only" if smoke_run else "formal_c2_frozen_input",
        "formal_c2_pack": not smoke_run,
        "selected_hidden_width": width,
        "split_id": split_id,
        "training_seed": training_seed,
        "time_steps": int(checkpoint["time_steps"]),
        "input_features": 800,
        "output_classes": 7,
        "structured_sparsity": config["training"]["structured_sparsity"],
        "weight_scales": scales,
        "weights_sha256": sha256_file(weight_path),
        "frozen_vectors_sha256": sha256_file(vector_path),
        "frozen_vector_count": int(len(test_index)),
        "board_required_for_physical_validation": True,
    }
    write_json(export_dir / "c2_pack_specification.json", specification)
    return specification


def summarize_decision_rules(split_metrics: list[dict[str, object]], config: dict[str, Any], bootstrap: dict[str, object]) -> dict[str, object]:
    macro_f1 = np.asarray([float(item["macro_f1"]) for item in split_metrics])
    ece = np.asarray([float(item["expected_calibration_error"]) for item in split_metrics])
    ordinal_mae = np.asarray([float(item["ordinal_step_mae"]) for item in split_metrics])
    within_one = np.asarray([float(item["errors_within_one_ordinal_step_fraction"]) for item in split_metrics])
    primary = next(item for item in split_metrics if item["split_id"] == config["splitting"]["primary_split_id"])
    return {
        "primary_readiness_passed": bool(
            float(primary["macro_f1"]) >= float(config["metrics"]["readiness_macro_f1_min"])
            and float(bootstrap["ci95_lower"]) >= float(config["metrics"]["readiness_ci95_lower_min"])
        ),
        "seed_instability_flagged": bool(macro_f1.std(ddof=1) > 0.03 or np.ptp(macro_f1) > 0.08),
        "calibration_instability_flagged": bool(float(primary["expected_calibration_error"]) > 0.10 or ece.std(ddof=1) > 0.03),
        "ordinal_interpretation_supported": bool(np.sum((ordinal_mae <= 1.0) & (within_one >= 0.70)) >= 4),
        "five_split_macro_f1": {
            "mean": float(macro_f1.mean()),
            "standard_deviation": float(macro_f1.std(ddof=1)),
            "minimum": float(macro_f1.min()),
            "maximum": float(macro_f1.max()),
        },
        "five_split_ece_standard_deviation": float(ece.std(ddof=1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one split, one seed, and two epochs for pipeline validation.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Result directory inside the repository root.")
    args = parser.parse_args()
    started = time.time()
    global RUN_RESULTS
    requested_results = args.results_dir or Path("outputs/spiking_smoke" if args.smoke else "outputs/spiking_training")
    RUN_RESULTS = (ROOT / requested_results).resolve() if not requested_results.is_absolute() else requested_results.resolve()
    if RUN_RESULTS != ROOT.resolve() and ROOT.resolve() not in RUN_RESULTS.parents:
        raise ValueError("results directory must remain inside the repository root")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    device = choose_device(args.device)
    maximum_epochs = int(args.max_epochs or config["training"]["maximum_epochs"])
    widths = [int(value) for value in config["model_selection"]["candidate_widths"]]
    split_ids = [f"split_{index:02d}" for index in range(1, 6)]
    training_seeds = [int(value) for value in config["model_selection"]["training_seeds"]]
    if args.smoke:
        split_ids = split_ids[:1]
        widths = widths[:1]
        training_seeds = training_seeds[:1]
        maximum_epochs = min(maximum_epochs, 2)

    RUN_RESULTS.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "data" / "source_responses.npz", allow_pickle=False) as archive:
        responses = archive["responses"].astype(np.float32)
    inputs, source_manifest = load_inputs()
    manifests = {split_id: load_split(split_id, source_manifest) for split_id in split_ids}

    validation_records: list[dict[str, object]] = []
    for split_id, metadata in manifests.items():
        for hidden_width in widths:
            for training_seed in training_seeds:
                print(f"TRAIN split={split_id} width={hidden_width} seed={training_seed}", flush=True)
                validation_records.append(
                    train_snn_candidate(
                        hidden_width=hidden_width,
                        split_id=split_id,
                        training_seed=training_seed,
                        responses=responses,
                        inputs=inputs,
                        metadata=metadata,
                        config=config,
                        device=device,
                        maximum_epochs=maximum_epochs,
                    )
                )
    validation_rows = pd.DataFrame(validation_records)
    validation_rows.to_csv(RUN_RESULTS / "model_selection_validation.csv", index=False, encoding="utf-8")
    selection = select_architecture(validation_rows, config)
    if args.smoke:
        selection["smoke_run"] = True
    write_json(RUN_RESULTS / "selected_architecture.json", selection)
    pd.DataFrame(selection["summary"]).to_csv(RUN_RESULTS / "model_selection_summary.csv", index=False, encoding="utf-8")

    snn_metrics, source_predictions, group_predictions = evaluate_selected_snn(
        selection, validation_rows, inputs, manifests, config, device
    )
    source_predictions.to_csv(RUN_RESULTS / "tloc_qsnn_source_predictions.csv", index=False, encoding="utf-8")
    group_predictions.to_csv(RUN_RESULTS / "tloc_qsnn_group_predictions.csv", index=False, encoding="utf-8")
    write_json(RUN_RESULTS / "tloc_qsnn_metrics.json", snn_metrics)
    pd.DataFrame([{key: value for key, value in item.items() if not isinstance(value, (list, dict))} for item in snn_metrics]).to_csv(
        RUN_RESULTS / "tloc_qsnn_metrics.csv", index=False, encoding="utf-8"
    )

    baseline_metrics, baseline_predictions = run_conventional_baselines(responses, inputs, manifests, config)
    write_json(RUN_RESULTS / "conventional_baseline_metrics.json", baseline_metrics)
    pd.DataFrame([{key: value for key, value in item.items() if not isinstance(value, (list, dict))} for item in baseline_metrics]).to_csv(
        RUN_RESULTS / "conventional_baseline_metrics.csv", index=False, encoding="utf-8"
    )
    baseline_predictions.to_csv(RUN_RESULTS / "conventional_baseline_group_predictions.csv", index=False, encoding="utf-8")

    primary_groups = group_predictions[group_predictions["split_id"] == config["splitting"]["primary_split_id"]]
    probability_columns = [f"probability_class_{class_id}" for class_id in range(7)]
    bootstrap = stratified_group_bootstrap_macro_f1(
        primary_groups["class_id"].to_numpy(dtype=np.int64),
        primary_groups[probability_columns].to_numpy(dtype=np.float64),
        resamples=100 if args.smoke else int(config["metrics"]["bootstrap_resamples"]),
        seed=20260806,
    )
    write_json(RUN_RESULTS / "primary_group_bootstrap.json", bootstrap)
    rules = summarize_decision_rules(snn_metrics, config, bootstrap)
    write_json(RUN_RESULTS / "predeclared_decision_rule_results.json", rules)
    c2_specification = export_c2_pack(selection, validation_rows, inputs, manifests, config, device, smoke_run=args.smoke)

    print(json.dumps({"selection": selection, "bootstrap": bootstrap, "decision_rules": rules}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
