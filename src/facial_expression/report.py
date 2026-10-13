"""Combine the two completed benchmark runs into a portfolio results overview."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ROOT
from .features import FEATURE_COLUMNS


def main():
    image = json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))
    grouped = json.loads((ROOT / "results/inferred_sequence/metrics.json").read_text(encoding="utf-8"))
    if image["metadata"]["mode"] != "full" or grouped["metadata"]["mode"] != "full":
        raise ValueError("Results overview requires both full benchmark runs")
    if image["metadata"]["dataset_sha256"] != grouped["metadata"]["dataset_sha256"]:
        raise ValueError("Comparison requires the same feature dataset")
    if image["metadata"]["split_strategy"] != "image" or grouped["metadata"]["split_strategy"] != "inferred-sequence":
        raise ValueError("Expected an image run in results/ and an inferred-sequence run in results/inferred_sequence/")
    if image["metadata"]["seed"] != grouped["metadata"]["seed"]:
        raise ValueError("Comparison requires the same random seed")
    if image["metadata"]["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Portfolio overview requires corrected feature schema version 2")
    seed = image["metadata"]["seed"]
    counts = image["metadata"]["split_counts"]
    group_counts = grouped["metadata"]["split_counts"]
    total = sum(counts.values())
    manifest = pd.read_csv(ROOT / "results/split_manifest.csv")
    if "inferred_frame_group" not in manifest or manifest["inferred_frame_group"].isna().any():
        raise ValueError("Overview requires legacy filenames with inferred frame groups")
    shared_groups = len(set(manifest.loc[manifest["split"] == "train", "inferred_frame_group"])
                        & set(manifest.loc[manifest["split"] == "test", "inferred_frame_group"]))
    names = ["dummy", "decision_tree", "svm", "mlp"]
    labels = ["Majority baseline", "Decision tree", "SVM", "PyTorch MLP"]
    image_f1 = [image["models"][name]["scores"]["test"]["macro_f1"] for name in names]
    group_f1 = [grouped["models"][name]["scores"]["test"]["macro_f1"] for name in names]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    positions = np.arange(len(names))
    a = ax.bar(positions - 0.18, image_f1, 0.36, label="Random image split", color="#2563eb")
    b = ax.bar(positions + 0.18, group_f1, 0.36, label="Inferred frame groups kept together", color="#0d9488")
    ax.bar_label(a, labels=[f"{v:.3f}" for v in image_f1], padding=4, fontsize=10)
    ax.bar_label(b, labels=[f"{v:.3f}" for v in group_f1], padding=4, fontsize=10)
    ax.set(xticks=positions, xticklabels=labels, ylim=(0, 1.15), ylabel="Test macro F1",
           title="Facial expression classification | evaluation sensitivity")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .025, f"{total} images | 25 corrected geometric features | seed {seed}\nFrame groups are inferred; neither protocol establishes performance on unseen people.",
             ha="center", fontsize=10, color="#475569")
    fig.tight_layout(rect=[0, .085, 1, 1])
    fig.savefig(ROOT / "results/evaluation_comparison.png", dpi=200)
    plt.close(fig)
    lines = ["# Reproduced benchmark results", "",
             "The portfolio benchmark compares classifiers using corrected feature schema version 2. "
             f"Both runs use the same {total} images and seed {seed}. All model selection is completed before test predictions are computed.", "",
             "![Comparison of evaluation strategies](../results/evaluation_comparison.png)", "",
             "| Model | Image split test macro F1 | Inferred-group test macro F1 |",
             "|---|---:|---:|"]
    for label, a, b in zip(labels, image_f1, group_f1):
        lines.append(f"| {label} | {a:.4f} | {b:.4f} |")
    lines += ["", "## What the comparison means", "",
              f"The image split contains {shared_groups} inferred frame groups shared between training and test. "
              "The grouped run keeps each candidate group together in both the outer split and cross-validation. "
              "It also tunes models independently using that protocol's training/validation data. "
              "The difference is a sensitivity comparison across protocols, not a causal estimate on identical test images.", "",
              "Groups are inferred from expression and filename index divided by three. "
              "They are not verified original sequence IDs and do not identify people. "
              "Neither score should be described as accuracy on unseen people or real-world emotion recognition. "
              "These are retrospective experiments on data already explored during the coursework; no fresh independent dataset was collected.", "",
              "## Reproduction", "", "```bash",
              "python -m facial_expression.features",
              f"python -m facial_expression.train --seed {seed} --output-dir results",
              f"python -m facial_expression.train --seed {seed} --split-strategy inferred-sequence --output-dir results/inferred_sequence",
              "python -m facial_expression.report", "```", "",
              "| Protocol | Train | Validation | Test | Classical CV |",
              "|---|---:|---:|---:|---|",
              f"| Stratified image | {counts['train']} | {counts['validation']} | {counts['test']} | Five stratified image folds |",
              f"| Inferred groups | {group_counts['train']} | {group_counts['validation']} | {group_counts['test']} | Five stratified group folds |", "",
              "Decision-tree and SVM searches maximize training cross-validation macro F1. "
              "Scaling and optional SMOTE are fitted separately in every fold. "
              "The MLP uses CPU training, 12 seeded Optuna trials (up to 200 epochs per trial), "
              "and validation macro F1 for tuning and stopping. "
              "Its final run allows 600 epochs and restores the best validation checkpoint. "
              "The MLP scaler and SMOTE use training rows only. "
              "Training scores use original images rather than synthetic samples. "
              "The validation metrics participate in model selection and are not independent performance estimates.", "",
              "## Evidence", "",
              "- [Image split detailed report](../results/report.md) and [metrics JSON](../results/metrics.json)",
              "- [Grouped detailed report](../results/inferred_sequence/report.md) and [metrics JSON](../results/inferred_sequence/metrics.json)",
              "- Split manifests, test predictions, search scores, per-class reports, and confusion matrices are saved beside each report.",
              "- [Feature provenance and limitations](data.md) and [historical coursework comparison](coursework.md)", "",
              f"Dataset SHA-256: `{image['metadata']['dataset_sha256']}`.", "",
              "Reproduction was verified locally on Python 3.12 using the package versions recorded in the metrics JSON files. "
              "PyTorch repeatability is targeted to the pinned CPU environment; numerical results can vary across platforms and dependency versions."]
    (ROOT / "docs/results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Generated docs/results.md and results/evaluation_comparison.png")


if __name__ == "__main__":
    main()
