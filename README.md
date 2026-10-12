# Eight-Channel Electronic-Nose Data and Code for Duplicate-Aware Ordered Mixture Screening

This repository contains electronic-nose measurements of Danggui (DG), Duhuo (DH), and their mixtures, together with code for duplicate-aware retrospective classification and integer/RTL inference checks. It accompanies **Duplicate-Aware Reliability Evaluation of an Eight-Channel Electronic-Nose System for Ordered Mixture Screening** by Xing Kang, Tong Yang, and Bin Ai.

## Data

The raw archive contains 550 measurement files: 50 files for each of 11 composition labels (pure DG, pure DH, and DG:DH ratios from 1:9 to 9:1). The classification benchmark uses 350 files from seven labels. Two pairs of byte-identical responses share group IDs, giving 348 evaluation groups. The group definition records file identity and exact duplication; it does not establish specimen, preparation-batch, session, date, or device independence.

The seven benchmark classes are DG, DG 1:9 DH, DG 3:7 DH, DG 5:5 DH, DG 7:3 DH, DG 9:1 DH, and DH. The four other mixture ratios remain available in the raw archive. The normalized dataset contains the 350 source IDs listed in its manifest.

Raw exports contain eight sensor columns and, in some files, two auxiliary columns. Acquisition was planned for 150 seconds. Short records were padded by repeating the last measured row. The supplied normalized responses have shape `350 × 150 × 8`; the benchmark extracts zero-based rows 30–129 and flattens each `100 × 8` window to 800 features. The original normalization implementation was not retained. The normalized files are therefore the starting point for reproducing the reported classification analysis; the release does not reconstruct normalization from raw exports.

## Repository contents

| Path | Contents |
|---|---|
| `data/raw/` | 550 original numeric exports, with file contents unchanged |
| `data/raw_manifest.csv` | Raw-file identities, composition labels, dimensions, and hashes |
| `data/normalized/` | 350 original normalized response files, named by stable source ID |
| `data/source_responses.npz` | Float32 response tensor in source-manifest order |
| `data/source_manifest.csv` | Source/group identities, raw-response links, class maps, and hashes |
| `data/outer_folds.csv` | Five outer test-fold assignments for the primary benchmark |
| `code/` | Nested classification, sensitivity, permutation, plotting, and inference checks |
| `config/` | Analysis seeds, candidate settings, and numeric representation |
| `results/` | Reference out-of-fold predictions, selections, metrics, and null distributions |
| `hardware/` | C1 fixed-point linear reference, C2 quantized spiking reference, and original FPGA user RTL/weights |

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for column definitions and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for all run commands.

## Quick start

Use Python 3.11. The reference analysis used NumPy 1.25.0, pandas 2.0.2, SciPy 1.11.4, and scikit-learn 1.3.2.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python code/verify_release.py
python code/run_benchmark.py
python code/compare_benchmark.py outputs/benchmark
```

Analysis commands write new results under `outputs/`. Reference files under `results/` remain unchanged.

## Reference findings

The primary analysis uses five outer and four inner stratified group folds. Standardization, logistic-regression regularization, and the probability-state selection are fitted within training partitions. Class probabilities are averaged within each group before calculating metrics, so each group receives one evaluation weight.

| Group-level quantity | Reference value |
|---|---:|
| Macro-F1 | 0.9943 |
| Class-stratified group bootstrap 95% interval (10,000 resamples) | 0.9857–1.0000 |
| Expected calibration error (10 bins) | 0.0106 |
| Multiclass Brier score | 0.0150 |
| Ordinal-step MAE | 0.0057 |
| Correct group decisions | 346/348 |

The sensitivity analyses rerun nested selection for shorter time windows, removal of individual sensor channels, and retention of one source per duplicate group. The refitting permutation analysis shuffles immutable-group labels 1,000 times and refits a fixed `C=100`, raw-probability model on the original outer folds. It does not repeat inner model selection. A separate frozen-decision permutation analysis tests label association without refitting.

C1 fixed-point agreement covers five linear models and 350 source-level vectors. C2 agreement covers 70 frozen vectors plus 280 deterministic coverage extensions; all 350 vectors are used for implementation checks rather than classifier-performance estimation. C2 uses a separate `800→64→7` quantized spiking model. It is distinct from both the C1 logistic models and the existing `800→7` binary-pulse FPGA prototype.

These findings apply to the retained archive. The data do not provide prospective external validation or measured board-level accuracy, power, or physical latency. DG and DH denote the supplied material labels; botanical authentication should not be inferred from filenames.

## Citation and license

Authors: Xing Kang, Tong Yang, and Bin Ai. School of Microelectronic and Communication Engineering, Chongqing University, Chongqing 400044, China. Contact: Bin Ai, binai@cqu.edu.cn; ORCID [0000-0002-0502-9166](https://orcid.org/0000-0002-0502-9166).

Use [CITATION.cff](CITATION.cff) to cite this version. Data, reference results, and documentation are licensed under [CC BY 4.0](LICENSE-DATA.md). Code and SystemVerilog are licensed under [MIT](LICENSE).
