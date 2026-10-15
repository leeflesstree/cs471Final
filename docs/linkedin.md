# LinkedIn project description

## Project title

**Facial Expression Classification with Geometric Features**

## Description

Compared SVM, decision-tree, and PyTorch MLP classifiers using 25 geometric facial features across seven expression classes in a CS 471 team project. My contributions included SVM implementation and hyperparameter tuning, data and feature engineering, visualizations, and code consolidation. I later improved the project with corrected feature calculations, a reproducible Python benchmark, cross-validation that keeps scaling and class balancing inside each training fold, and automated reports, assisted by Codex. I also compared image-level evaluation with an inferred-group sensitivity analysis to investigate possible dependence between related frames. The evaluation is limited to a small CK+-derived dataset and does not establish generalization to unseen people.

## Short version

Built and tuned an SVM for facial-expression classification using engineered landmark features as part of a three-person project. Contributed feature engineering, visualizations, and code consolidation, then added feature corrections, a reproducible benchmark, and a sensitivity analysis of the evaluation splits with Codex assistance.

## Suggested skills

Python · Machine Learning · Feature Engineering · scikit-learn · Support Vector Machines · PyTorch · Model Evaluation · Data Visualization

## Featured attachment

Link to the repository and use a generated model-comparison figure as the preview image. The [results report](results.md) compares the two evaluation strategies and provides the metrics and protocol to explain in an interview.

If you add a score to this description, take it from a reproduced **test macro F1** column in the results report and name the evaluation strategy. Include both strategies when discussing the effect of related frames. The report, presentation, and original notebook contain historical numbers from different runs.

## Optional announcement post

I revisited my CS 471 team project on facial-expression classification and made the experiments reproducible.

We compared a decision tree, an SVM, and a PyTorch MLP using 25 features derived from facial landmarks. My original work focused on the SVM, feature engineering, visualizations, and bringing the code together. The updated version adds corrected feature calculations, a reproducible benchmark, preprocessing and SMOTE inside cross-validation, and generated metrics and figures, with Codex assistance.

In the revised benchmark, SVM test macro F1 was 0.990 with random image splitting and 0.623 when inferred frame groups stayed together.

One lesson from the project: the evaluation setup matters as much as the model. I compared an image-level split with a sensitivity analysis that keeps likely groups of related frames together. Those groups are inferred from filenames and feature similarity; evaluation on entirely new people still requires reliable subject identities.

Project: https://github.com/leeflesstree/cs471Final

Team: Alexander Nicholas, Kieran Hawkins, and Michael Duggan.

## GitHub presentation suggestions

**Description:** Facial-expression classification using 25 geometric features, SVM, decision trees, and PyTorch, with reproducible evaluation and visual reports.

**Topics:** `machine-learning`, `facial-expression-recognition`, `feature-engineering`, `scikit-learn`, `pytorch`, `svm`, `classification`, `computer-vision`

The display title works with the current repository URL. A repository rename is optional and has not been performed as part of the documentation.
