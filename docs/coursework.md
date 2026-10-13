# Coursework history, provenance, and contributions

## Original project

The original project, **Emotion Detection Using Facial Landmarks**, was submitted for **CS 471: Artificial Intelligence** on **May 9, 2025**. The authors are **Alexander Nicholas, Kieran Hawkins, and Michael Duggan**.

The report describes an experiment using the Extended Cohn-Kanade (CK+) dataset and 25 engineered features extracted from MediaPipe Face Mesh landmarks. The seven labels in the supplied feature table are `anger`, `contempt`, `disgust`, `fear`, `happy`, `sadness`, and `surprise`.

The earlier GitHub README described FER-2013, 68 landmarks, and logistic regression. The final coursework materials instead document CK+, MediaPipe's 468 landmarks, SVM, decision tree, and a PyTorch MLP. This version follows the final materials.

The [coursework archive](../archive/coursework/) preserves the original code, consolidated notebook, report, slides, and supplied feature data. The repository state preceding portfolio improvements is commit [`8af61bf2500068bb36cf63f464121cbf08d4e5db`](https://github.com/leeflesstree/cs471Final/commit/8af61bf2500068bb36cf63f464121cbf08d4e5db).

## Team contributions

These roles come from the final report's member contribution section:

| Team member | Original work |
| --- | --- |
| Alexander Nicholas | Implemented the SVM in scikit-learn and tuned it with GridSearchCV; contributed data/feature engineering, data visualization, code consolidation, report, and slides |
| Kieran Hawkins | Implemented the MLP in PyTorch and tuned it with Optuna; contributed report and slides |
| Michael Duggan | Implemented the decision tree and tuned it with GridSearchCV; contributed report and slides |

The later portfolio work was carried out by Alexander with Codex assistance. It corrects feature calculations, packages the benchmark, improves reproducibility and evaluation, adds an inferred-group sensitivity analysis, produces consistent results reports, and updates the documentation. Original model and team contributions retain the attribution above.

## Data provenance and evaluation scope

The supplied coursework `features.csv` has **981 rows**, with **25 numeric features**, an image filename, and an expression label. It is preserved byte-for-byte as [data/features_original.csv](../data/features_original.csv). The primary [data/features.csv](../data/features.csv) is regenerated from all 981 committed landmark rows with the corrections described below and in [docs/data.md](data.md). The coursework identifies the data as derived from CK+ and cites [this Kaggle dataset page](https://www.kaggle.com/datasets/shuvoalok/ck-dataset/data).

| Label | Samples |
| --- | ---: |
| anger | 135 |
| contempt | 54 |
| disgust | 177 |
| fear | 75 |
| happy | 207 |
| sadness | 84 |
| surprise | 249 |

Image filenames in the supplied table do not provide a verified mapping to original subject or sequence identifiers. A filename or feature duplicate check cannot establish subject independence. The default portfolio benchmark therefore documents its image-level split explicitly.

The archive includes supplied train/validation/test files with **685/148/148** rows. The consolidated coursework notebook instead splits the full table and reports **686/147/148** rows. The default portfolio benchmark regenerates a fixed stratified split from the full table with seed `42`, producing **686/147/148** rows. This avoids treating conflicting split files as interchangeable.

The second evaluation keeps inferred groups of related frames together. Filenames follow `Ck+_<emotion>_<index>.png`; grouping by expression label and `index // 3` gives **327 groups of three**. A nearest-neighbor check supports investigating dependence within these triples; [docs/data.md](data.md) documents the check and its limitations. This does not verify the original sequence identities. The grouped sensitivity analysis keeps those groups together in both the data split and cross-validation, producing **684/147/150** training/validation/test rows. Neither strategy establishes subject-disjoint evaluation.

Raw source images are absent. The current benchmark runs from features regenerated from stored landmarks, so the full image-to-prediction path has not been independently reproduced. Further work on subject-disjoint evaluation or deployment requires the source images, appropriate dataset access, and reliable subject/sequence metadata. Dataset reuse remains subject to the original source's terms.

## Historical results

The values below are saved coursework outputs, provided for traceability. Use the [generated results report](results.md) for the portfolio benchmark.

| Model | Consolidated notebook: test macro F1 | Final report: test F1 |
| --- | ---: | ---: |
| Decision tree | 0.8317 | approximately 0.83 |
| SVM | 0.9897 | 0.9897 |
| MLP | 0.9626 | 0.9468 |

The notebook, report, and slides describe different MLP settings and results. In addition, the original hyperparameter searches receive data that have already been scaled and resampled. Doing that before cross-validation lets information from validation folds influence the preprocessing or synthetic samples used for training. The portfolio benchmark fits these operations inside each training fold.

The consolidated notebook resolves several problems present in earlier model files: it trains the decision tree on the training data, applies Optuna's sampled MLP settings, and evaluates the MLP tuning objective across the validation set. Those fixes should be attributed to the consolidated coursework version rather than the later portfolio cleanup.

## Feature interpretation

The original feature extractor and report do not define every column identically. For example, the extractor divides aspect ratios by the interocular normalizer a second time, and its four symmetry columns represent raw horizontal landmark differences rather than all of the formulas printed in the report. Several cited landmark indices also differ. The archive retains those artifacts so their historical results can be interpreted in context.

The portfolio feature extractor removes the second normalization of aspect ratios, normalizes symmetry offsets, aligns symmetry column names with their calculations, and uses eye corners `33` and `263` consistently for the eye-line angle. Tests check the corrected representation's translation and scale invariance. [docs/data.md](data.md) describes the formulas and regeneration process.

The corrected primary dataset and byte-for-byte original table are stored separately. Changing feature definitions changes the experiment, so the reproduced portfolio results should not be presented as a rerun of the historical feature representation.

The historical feature selection section is titled "Top 10," while the consolidated notebook uses `k=15`. This inconsistency is preserved in the archive rather than repeated as a benchmark claim.

## Scope of the portfolio version

- A Python package and notebook entry point replace personal Colab Drive paths.
- Corrected geometric features are regenerated from all stored landmarks, with the original table preserved separately.
- A fixed split and recorded seeds make the experiment reproducible.
- Model selection keeps scaling and SMOTE inside cross-validation.
- An inferred-group sensitivity analysis investigates likely related frames while documenting the absence of verified sequence and subject identities.
- A dummy baseline, macro F1, accuracy, and confusion matrices make the comparison easier to assess.
- Generated reports and figures provide one consistent source of benchmark results.

The original report and slides remain historical deliverables. They are useful evidence of the team's process and contributions, and their saved numbers should be read with the limitations above.
