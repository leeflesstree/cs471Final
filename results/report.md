# Benchmark results

Split strategy: `image`. Seed: 42.

| Model | Train macro F1 | Validation macro F1 | Test macro F1 | Test accuracy |
|---|---:|---:|---:|---:|
| Majority baseline | 0.0578 | 0.0575 | 0.0584 | 0.2568 |
| Decision tree | 1.0000 | 0.9102 | 0.8105 | 0.8716 |
| SVM | 1.0000 | 0.9903 | 0.9897 | 0.9932 |
| PyTorch MLP | 1.0000 | 0.9903 | 0.9632 | 0.9730 |

Training metrics use original training images, never synthetic SMOTE rows.

See `metrics.json` for selected parameters, dataset hash, dependency versions, and per-class reports.

These retrospective results do not establish generalization to new subjects. Inferred frame groups are unverified.
