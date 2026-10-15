# Benchmark results

Split strategy: `inferred-sequence`. Seed: 42.

| Model | Train macro F1 | Validation macro F1 | Test macro F1 | Test accuracy |
|---|---:|---:|---:|---:|
| Majority baseline | 0.0579 | 0.0562 | 0.0590 | 0.2600 |
| Decision tree | 0.8669 | 0.5050 | 0.4664 | 0.6333 |
| SVM | 0.8832 | 0.6710 | 0.6230 | 0.7400 |
| PyTorch MLP | 0.9841 | 0.6952 | 0.6570 | 0.7600 |

Training metrics use original training images, never synthetic SMOTE rows.

See `metrics.json` for selected parameters, dataset hash, dependency versions, and per-class reports.

These retrospective results do not establish generalization to new subjects. Inferred frame groups are unverified.
