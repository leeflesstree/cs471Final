# Data, feature corrections, and evaluation limits

This project compares classifiers on seven CK+ facial-expression labels using 25 geometric features derived from MediaPipe's 468 x/y landmarks. The available exports contain 981 images: 135 anger, 54 contempt, 177 disgust, 75 fear, 207 happy, 84 sadness, and 249 surprise. The repository does not include source face images, a verified original-image manifest, or subject IDs.

## Verified provenance

`data/features_original.csv` preserves the complete feature table supplied with the final assignment. The archived GitHub landmark exports contain the same 981 unique image names and the same expression labels. Recomputing the coursework formulas from those landmarks reproduces every original feature to floating-point precision: the maximum absolute residual in the initial inspection was 1.78 × 10⁻¹⁵. The regeneration command repeats this check and records its actual residual in `data/provenance.json`.

The original feature table has no missing or nonfinite values and no exact duplicate feature vectors. These checks do not rule out closely related images. `image_name` contains the expression label and is used only as a row identifier; it must never be included as a model feature.

Regenerate corrected features with:

```bash
python -m facial_expression.features
```

The command reads the two archived landmark CSVs, verifies names, labels, and reproduction of the historical features, and aligns generated rows with `features_original.csv`. It saves corrected features to `data/features.csv`. `data/provenance.json` records source and output SHA-256 hashes, class counts, the ordered feature schema, and the formula changes. It deliberately fails if the reference table cannot be reproduced by the available landmarks.

## Feature schema version 2

The historical extractor introduced several inconsistencies:

- Eye and mouth height/width ratios were divided by the outer-eye distance again, making already dimensionless ratios depend on face size.
- Four horizontal symmetry offsets were left unnormalized, while most other lengths were normalized.
- The last two symmetry-column names did not match their landmark pairs: `(159, 386)` uses upper-eye points; `(105, 334)` uses eyebrow points.
- The documented outer-eye angle used one outer corner and one inner corner, `(33, 362)`.

Version 2 normalizes distance-like features and all four symmetry offsets by the outer-eye span `(33, 263)`, retains dimensionless aspect ratios, and computes the eye angle from the two outer corners. The last four columns are `eye_symmetry`, `mouth_symmetry`, `upper_eye_symmetry`, and `eyebrow_symmetry`. Both angle features use degrees. Zero or near-zero denominators raise an error rather than adding a scale-dependent epsilon. Tests verify invariance under translation and positive uniform scaling, the ratio definitions, geometry validation, and full-dataset regeneration.

These features retain image orientation. They do not remove head rotation, pose effects, or facial identity. The landmark exports contain normalized x/y coordinates without original image dimensions, so the pipeline also cannot correct distances for unknown image aspect ratios. Corrected results use schema version 2 and should not be presented as direct reproductions of the historical report's scores.

## Splits and related frames

The final-assignment export supplied 685/148/148 training/validation/test rows, whose union equals the full 981-row dataset. Its consolidated notebook instead performs a fresh 70/15/15 stratified image split from the full table, producing 686/147/148 rows. These are different experiments. The current benchmark's saved split manifests and run configuration define which rows each result actually uses; comparisons must use the same protocol and schema.

Renamed files use `Ck+_<expression>_<integer>.png`, with no verified subject or sequence identifier. An inspection found a strong pattern: grouping each expression's numeric suffix by integer division by three yields 327 groups, each containing exactly three images. In the supplied historical split, 216 of those groups appear in more than one split. Across the full historical feature table, 99.59% of images have their nearest neighbor in standardized feature space within the same candidate group.

This is evidence of closely related frames crossing the historical split. It does **not** prove that these inferred groups are original CK+ sequences, and it cannot identify subjects appearing in multiple groups. If an experiment keeps these candidate groups together, describe it as an **inferred adjacent-frame grouping sensitivity analysis**. Neither that experiment nor a random image split establishes performance on unseen people.

Verified subject-disjoint evaluation requires original CK+ filenames or a trustworthy mapping from renamed files to subject and sequence IDs. A future preprocessing run should retain those IDs before splitting. The available archive supports a reproducible feature benchmark, with that generalization limit stated explicitly.

## Interpreting results

Macro F1 is the main comparison metric because it gives equal weight to each expression class. Accuracy, weighted F1, per-class support and F1, and confusion matrices provide context. Feature scaling, feature selection, and any resampling must be fitted within each training fold; validation and test examples must retain their original distribution. Training-set metrics should also use real training rows rather than synthetic SMOTE samples.

The archived notebook and report already display held-out scores across many exploratory model variants. Their historical test data is therefore not an untouched prospective benchmark. New runs make the procedure reproducible and correct specific implementation issues; they do not establish real-world or person-independent expression recognition. The task is classification of dataset expression labels, not measurement of a person's internal emotional state.

## Method references

- [imbalanced-learn: common pitfalls and pipelines](https://imbalanced-learn.org/stable/common_pitfalls.html)
- [scikit-learn: cross-validation and grouped data](https://scikit-learn.org/stable/modules/cross_validation.html)
- [PyTorch: reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
