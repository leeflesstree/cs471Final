"""Run the complete benchmark: python -m facial_expression.train."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            ConfusionMatrixDisplay, f1_score)

from .data import CLASSES, ROOT, inferred_groups, load_data, make_splits, xy
from .models import make_search
from .neural import tune_mlp

DISPLAY = {"dummy": "Majority baseline", "decision_tree": "Decision tree", "svm": "SVM", "mlp": "PyTorch MLP"}


def summarize_params(params):
    return {key: (value if isinstance(value, (str, int, float, bool, type(None))) else repr(value))
            for key, value in params.items()}


def evaluate(model, df):
    X, y = xy(df)
    pred = model.predict(X)
    report = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    scores = {"accuracy": accuracy_score(y, pred),
              "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
              "weighted_f1": f1_score(y, pred, labels=CLASSES, average="weighted", zero_division=0)}
    return scores, report, pred


def plot_confusion(y, pred, path, title):
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=CLASSES), display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=40
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_plots(metrics, splits, output):
    fig, ax = plt.subplots(figsize=(9, 5))
    names = ["dummy", "decision_tree", "svm", "mlp"]
    values = [metrics[name]["scores"]["test"]["macro_f1"] for name in names]
    bars = ax.bar([DISPLAY[name] for name in names], values, color=["#94a3b8", "#0d9488", "#2563eb", "#8b5cf6"])
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4)
    ax.set(ylim=(0, 1.08), ylabel="Test macro F1", title="Facial expression classification")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=180)
    plt.close(fig)
    counts = pd.DataFrame({name: part["emotion"].value_counts().reindex(CLASSES)
                           for name, part in splits.items()})
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot.bar(ax=ax, color=["#2563eb", "#0d9488", "#f59e0b"], rot=25)
    ax.set(title="Original class distribution by split", ylabel="Images", xlabel="Expression label")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=180)
    plt.close(fig)


def run(data_path, output, seed=42, strategy="image", quick=False):
    started = time.perf_counter()
    df = load_data(data_path)
    splits = make_splits(df, seed, strategy)
    output.mkdir(parents=True, exist_ok=True)
    # Manifest captures membership, not duplicated features or synthetic training rows.
    manifest = pd.concat([part[["image_name", "emotion"]].assign(split=name)
                          for name, part in splits.items()])
    try:
        manifest["inferred_frame_group"] = inferred_groups(manifest)
    except ValueError:
        # Image-level evaluation also supports feature CSVs without the legacy naming convention.
        if strategy == "inferred-sequence":
            raise
    manifest.to_csv(output / "split_manifest.csv", index=False)
    X_train, y_train = xy(splits["train"])
    X_val, y_val = xy(splits["validation"])
    metadata = {
        "seed": seed, "mode": "quick_smoke" if quick else "full",
        "mlp_training": {"device": "cpu", "trial_epoch_limit": 30 if quick else 200,
                         "trial_patience": 10 if quick else 40,
                         "final_epoch_limit": 30 if quick else 600,
                         "final_patience": 10 if quick else 60},
        "split_strategy": strategy, "split_counts": {k: len(v) for k, v in splits.items()},
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "feature_columns": list(df.columns[1:-1]), "classes": CLASSES,
        "python": platform.python_version(), "platform": platform.system(),
        "versions": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "optuna", "torch", "matplotlib"]},
        "limitations": ["Image-level data have no verified subject or sequence identifiers.",
                        "Inferred frame groups use emotion and index//3; they are a sensitivity analysis, not verified sequences or subjects.",
                        "Historical test data have already been inspected; these are retrospective benchmark results.",
                        "Single fixed seed; no independent dataset or real-world emotion inference validation."],
    }
    metrics = {}
    models = {}
    selection = {}
    print(f"Split: {strategy}, counts={metadata['split_counts']}, seed={seed}", flush=True)
    models["dummy"] = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    selection["dummy"] = {"parameters": {"strategy": "most_frequent"}}
    for name in ("decision_tree", "svm"):
        print(f"Tuning {name} with training-only cross-validation...", flush=True)
        search = make_search(name, seed, quick, grouped=strategy == "inferred-sequence")
        groups = inferred_groups(splits["train"]) if strategy == "inferred-sequence" else None
        search.fit(X_train, y_train, groups=groups)
        models[name] = search.best_estimator_
        selection[name] = {"parameters": summarize_params(search.best_params_),
                           "training_cv_macro_f1": search.best_score_}
        pd.DataFrame({key: value for key, value in search.cv_results_.items()
                      if key in ["params", "mean_test_score", "std_test_score", "rank_test_score"]}).to_csv(
            output / f"tuning_{name}.csv", index=False
        )
    print("Tuning seeded PyTorch MLP on validation macro F1...", flush=True)
    models["mlp"], selection["mlp"], history = tune_mlp(X_train, y_train, X_val, y_val, seed, quick)
    pd.DataFrame(history).to_csv(output / "mlp_history.csv", index=False)
    # Selection is finished for every model before any test prediction is computed.
    rows = []
    for name, model in models.items():
        metrics[name] = {"selection": selection[name], "scores": {}, "reports": {}}
        for split_name, part in splits.items():
            scores, report, pred = evaluate(model, part)
            metrics[name]["scores"][split_name] = scores
            metrics[name]["reports"][split_name] = report
            rows.append({"model": name, "split": split_name, **scores})
            if split_name == "test":
                pd.DataFrame({"image_name": part["image_name"].to_numpy(), "true_label": part["emotion"].to_numpy(),
                              "predicted_label": pred}).to_csv(output / f"predictions_{name}.csv", index=False)
                plot_confusion(part["emotion"], pred, output / f"confusion_{name}.png",
                               f"{DISPLAY[name]} | {strategy} | test set")
        print(f"{name}: test macro F1={metrics[name]['scores']['test']['macro_f1']:.4f}", flush=True)
    importance = permutation_importance(models["svm"], X_val, y_val, scoring="f1_macro",
                                         n_repeats=5 if quick else 10, random_state=seed)
    imp = pd.DataFrame({"feature": df.columns[1:-1], "mean_macro_f1_drop": importance.importances_mean,
                        "std": importance.importances_std}).sort_values("mean_macro_f1_drop", ascending=False)
    imp.to_csv(output / "svm_validation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    top = imp.head(10).iloc[::-1]
    ax.barh(top["feature"], top["mean_macro_f1_drop"], xerr=top["std"], color="#2563eb")
    ax.set(title="SVM permutation importance on validation data", xlabel="Decrease in macro F1")
    fig.tight_layout()
    fig.savefig(output / "feature_importance.png", dpi=180)
    plt.close(fig)
    metadata["duration_seconds"] = time.perf_counter() - started
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    (output / "metrics.json").write_text(json.dumps({"metadata": metadata, "models": metrics}, indent=2), encoding="utf-8")
    write_plots(metrics, splits, output)
    lines = [f"# {'Smoke check' if quick else 'Benchmark'} results", "", f"Split strategy: `{strategy}`. Seed: {seed}.", "",
             "| Model | Train macro F1 | Validation macro F1 | Test macro F1 | Test accuracy |",
             "|---|---:|---:|---:|---:|"]
    for name in models:
        s = metrics[name]["scores"]
        lines.append(f"| {DISPLAY[name]} | {s['train']['macro_f1']:.4f} | {s['validation']['macro_f1']:.4f} | {s['test']['macro_f1']:.4f} | {s['test']['accuracy']:.4f} |")
    lines += ["", "Training metrics use original training images, never synthetic SMOTE rows.", "",
              "See `metrics.json` for selected parameters, dataset hash, dependency versions, and per-class reports.", "",
              "These retrospective results do not establish generalization to new subjects. Inferred frame groups are unverified."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"metadata": metadata, "models": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/features.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split-strategy", choices=["image", "inferred-sequence"], default="image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Small smoke run; results are not the portfolio benchmark")
    args = parser.parse_args()
    output = args.output_dir or (ROOT / "results" / ("quick" if args.quick else "inferred_sequence" if args.split_strategy == "inferred-sequence" else ""))
    if args.quick and output.resolve() == (ROOT / "results").resolve():
        parser.error("Quick mode must use a separate output directory")
    run(args.data, output, args.seed, args.split_strategy, args.quick)


if __name__ == "__main__":
    main()
