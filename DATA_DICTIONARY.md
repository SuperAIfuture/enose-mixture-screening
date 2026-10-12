# Data dictionary

## Raw exports

Files are comma-delimited numeric text without a header. Rows follow the recorded time order. Most records contain 150 rows; original shorter records are preserved. Columns 1–8 are the eight sensor channels. Ten-column files additionally contain the temperature/humidity auxiliary pair. The retained exports do not establish the order or units of the auxiliary pair or a complete sensor-channel-to-part-number/unit mapping. Raw numeric values are supplied in their native recorded form, without assigning an inferred physical unit. Eight-column files contain no auxiliary columns.

`data/raw_manifest.csv` defines `raw_id`, DG/DH ratio parts, original label and filename, repository-relative `raw_path`, actual `rows` and `columns`, SHA-256, and `included_in_benchmark`. A ratio label describes the prepared-material composition recorded for a file; it is not a measured concentration or continuous regression target.

## Normalized responses

Each `data/normalized/<source_id>.txt` contains 150 rows and eight comma-delimited, dimensionless response columns. `data/source_responses.npz` contains one key, `responses`, with a float32 array of shape `(350,150,8)`. Array index order matches `data/source_manifest.csv`. Text files retain their original numeric precision and bytes; the tensor stores their float32 interpretation.

| Source-manifest field | Definition |
|---|---|
| `source_id` | Stable source ID; `cNN_sNNN`, zero-based within-class archive order |
| `class_id`, `class_label` | Archive class encoding and readable label |
| `within_class_index` | Zero-based within-class position |
| `ordinal_index` | Position on the monotonic selected-ratio order |
| `hardware_class_id` | Encoding used by the original binary-pulse FPGA prototype |
| `group_id` | File identity or same-class byte-duplicate group |
| `group_size`, `group_members` | Number and semicolon-separated source IDs in the group |
| `raw_id`, `raw_path` | Matching raw-export identifier and relative path |
| `response_path` | Relative path to the original normalized text file |
| `response_sha256`, `raw_sha256` | SHA-256 of the corresponding unchanged file bytes |
| `rows`, `columns`, `minimum`, `maximum` | Shape and float32 numeric range of the response |

| Class ID | Label | Ordinal position | Original hardware ID |
|---|---|---:|---:|
| 0 | DG | 0 | 0 |
| 1 | DG 1:9 DH | 5 | 1 |
| 2 | DG 3:7 DH | 4 | 3 |
| 3 | DG 5:5 DH | 3 | 2 |
| 4 | DG 7:3 DH | 2 | 4 |
| 5 | DG 9:1 DH | 1 | 5 |
| 6 | DH | 6 | 6 |

The selected monotonic ratio order is DG, DG 9:1 DH, DG 7:3 DH, DG 5:5 DH, DG 3:7 DH, DG 1:9 DH, DH. Adjacent ordinal steps are not all equally spaced in mixture fraction. C1 and C2 reference decisions use archive class IDs; the original hardware map applies only to that separate prototype.

The duplicate pairs originate from DG 7:3 DH raw filenames `混合-14.txt`/`混合-20.txt` and `混合-39.txt`/`混合-48.txt`. Group IDs describe exact duplication, not biological independence. No specimen, batch, session/date, or device identifiers are supplied.

## Reference predictions and metrics

Source-level OOF tables provide `source_id`, `group_id`, true `class_id`, `outer_fold` (1–5), `predicted_class_id`, and `probability_class_0` through `probability_class_6`. Group-level tables replace source identity with `source_count`. Group probabilities are within-group arithmetic means. Predicted class is the lowest-index argmax in a tie. `data/outer_folds.csv` records each source's one outer test-fold assignment; its training partition is the complement.

Inner selection tables report candidate `c`, `probability_state`, mean inner-fold macro-F1/Brier/ECE, and `selected`. Temperature scaling is fitted on the corresponding training probabilities. Metrics use group-weighted decisions, ten equal-width confidence bins for ECE, the sum-over-classes multiclass Brier score, and the ordinal map above. Bootstrap samples groups with replacement separately within each true class.

## Hardware references

C1 memory files store 12-bit unsigned inputs (scale 4095), 24-bit signed weights, and 64-bit signed biases/accumulators as hexadecimal words; weight fractional bits are recorded per fold. Features use row-major time/channel order. The model scores each of seven classes and selects the lowest-index maximum.

`hardware/c2/vectors/bridge_full_vectors.npz` stores `frames` `(350,30,800)` uint8, `counts` `(350,7)` int16, `predictions` `(350,)`, and source/group/label arrays. Frame bit order is little-endian within each packed 100-byte vector. `source_ledger.csv` distinguishes `frozen_split_01` from `deterministic_source_seed_extension`. Count/decision agreement measures implementation conformance. Label arrays are source metadata and are not a second performance endpoint.
