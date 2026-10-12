"""Integer LIF reference and deterministic coverage-vector encoding."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

def load_signed_int8_mem(path: Path, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(
        [int(line, 16) for line in path.read_text(encoding="utf-8").splitlines() if line],
        dtype=np.int16,
    )
    values[values >= 128] -= 256
    return values.astype(np.int8).reshape(shape)

def fixed_update(membrane_q: np.ndarray, dot: np.ndarray, coefficient_q: int) -> np.ndarray:
    numerator = membrane_q.astype(np.int64) * 1024 + 2 * dot.astype(np.int64) * coefficient_q
    sign = np.where(numerator < 0, -1, 1).astype(np.int64)
    absolute = np.abs(numerator)
    quotient, remainder = np.divmod(absolute, 2048)
    round_up = (remainder > 1024) | ((remainder == 1024) & ((quotient & 1) == 1))
    return sign * (quotient + round_up.astype(np.int64))

def run_integer_reference(
    frames: np.ndarray, hidden_weight: np.ndarray, output_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    counts: list[np.ndarray] = []
    max_hidden = 0
    max_output = 0
    hidden_weight_i64 = hidden_weight.astype(np.int64)
    output_weight_i64 = output_weight.astype(np.int64)
    for sample in frames.astype(np.int64):
        hidden_membrane = np.zeros(64, dtype=np.int64)
        output_membrane = np.zeros(7, dtype=np.int64)
        output_count = np.zeros(7, dtype=np.int64)
        for frame in sample:
            hidden_dot = frame @ hidden_weight_i64.T
            hidden_next = fixed_update(hidden_membrane, hidden_dot, 928)
            hidden_spike = hidden_next > 128
            hidden_membrane = np.where(hidden_spike, 0, hidden_next)
            max_hidden = max(max_hidden, int(np.abs(hidden_membrane).max()))

            output_dot = hidden_spike.astype(np.int64) @ output_weight_i64.T
            output_next = fixed_update(output_membrane, output_dot, 1142)
            output_spike = output_next > 128
            output_membrane = np.where(output_spike, 0, output_next)
            max_output = max(max_output, int(np.abs(output_membrane).max()))
            output_count += output_spike.astype(np.int64)
        counts.append(output_count)
    count_array = np.stack(counts).astype(np.int16)
    return count_array, count_array.argmax(axis=1).astype(np.int16), max_hidden, max_output

def source_seed(source_id: str) -> int:
    digest = hashlib.sha256(f"c1-c2-bridge-v1:{source_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)

def encode_extension_frames(features: np.ndarray, source_id: str, probability_scale: float) -> np.ndarray:
    probability = np.clip(features * probability_scale, 0.0, 1.0)
    rng = np.random.default_rng(source_seed(source_id))
    return (rng.random((30, 800)) < probability[None, :]).astype(np.uint8)

def frame_to_hex(frame: np.ndarray) -> str:
    packed = np.packbits(frame.astype(np.uint8), bitorder="little")
    value = int.from_bytes(packed.tobytes(), "little")
    return f"{value:0200X}"
