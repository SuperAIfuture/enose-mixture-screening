# Reproducing the analyses

Run commands from the repository root after installing `requirements.txt` in Python 3.11. The scripts resolve input files relative to their own location. Numerical libraries and thread settings can change fitted floating-point probabilities. The reference predictions remain available for exact metric reconstruction. Comparing a fresh fit reports both decision agreement and probability differences.

## Integrity and primary benchmark

```sh
python code/verify_release.py
python code/run_benchmark.py --output outputs/benchmark
python code/compare_benchmark.py outputs/benchmark
```

The verifier checks file hashes, all 550 raw shapes, 350 normalized files against the response tensor, duplicate groups, five outer folds and four inner folds, OOF aggregation, all recorded primary metrics, and the 10,000-resample bootstrap. The benchmark independently regenerates the nested folds and fits all candidates; its output does not replace the reference results.

## Sensitivity analyses

```sh
python code/run_sensitivity.py --workers 2
# A single condition:
python code/run_sensitivity.py --condition time_t040_all8 --workers 2
```

The default runs 13 nested conditions: a 100-point control, 40/60/80-point windows, eight leave-one-channel-out analyses, and a one-source-per-group analysis. Each condition repeats inner selection. Outputs include source/group predictions and the selected settings per outer fold.

## Permutation analyses

```sh
python code/run_permutations.py --mode association --permutations 1000
python code/run_permutations.py --mode refit --permutations 1000 --workers 2
```

Both analyses permute labels once per immutable group using seed base 2026080711 and reuse the five outer folds. The association analysis fixes the observed OOF decisions. The refit analysis trains a `C=100`, raw-probability classifier for each outer fold under each permuted group-label assignment. Neither repeats inner hyperparameter/probability-state selection. The empirical p-value is `(1 + exceedances)/(1 + permutations)`. Refitting 1,000 permutations is substantially more expensive than reconstructing the stored null summaries.

## Integer and RTL inference

```sh
python code/verify_hardware.py
python code/export_fixed_point.py
python code/run_rtl.py --iverilog iverilog --vvp vvp
```

Install Icarus Verilog 12 separately for RTL execution. The integer verifier recomputes all five C1 fixed-point memories and C2 spike counts/decisions. The RTL runner copies memories to a temporary work directory, compiles the supplied cores, and checks the literal PASS markers in simulator output. C2 exposes seven 16-bit counts as a packed port for Icarus compatibility. This preserves the arithmetic and bit widths.

C1 fixed-point and C2 quantized-spiking references are independent implementation paths. These checks provide no board-level accuracy, measured power, physical latency, synthesis, or place-and-route results. The frozen C2 weights and frames are supplied for inference/conformance reproduction.

## Secondary spiking-model training and original FPGA code

```sh
python -m pip install -r requirements-spiking.txt
python code/run_spiking_training.py --device cpu --smoke
# Full candidate-width/seed search on five fixed group partitions:
python code/run_spiking_training.py --device cpu
```

The secondary training pipeline uses five group-preserving train/validation/test partitions under `data/spiking_splits/`; these are distinct from the primary C1 nested outer folds. The smoke command runs one split, one seed, one candidate width, and two epochs, and checks the training/export path only. Full training searches widths 0/32/64/128 using three seeds and selects by validation results. Its performance is not the primary 348-group C1 result. The frozen C2 int8 model is supplied independently of a new training run.

`hardware/fpga_prototype/` contains the 12 original user HDL modules, seven ROM COE files, LCD bitmap data, and original board constraints. See its README for the device, clock, IP dependencies, and distinction from the two inference references. The original HDL comments were converted from GBK to UTF-8 in the published copies; the logic remains unchanged.

## Plotting

```sh
python code/plot_results.py
```

This command creates a group-level confusion matrix, a representative measured response plot, a sensitivity summary, and permutation distributions under `outputs/figures/`.

## Version and citation

Version 1.0.0 contains raw measurements, the selected normalized responses, reference statistical results, and portable analysis/inference code. Use `CITATION.cff`. Data/result hashes are recorded in `SHA256SUMS`; reruns go under the ignored `outputs/` directory.
