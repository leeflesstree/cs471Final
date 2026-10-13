"""Regression checks for split independence, preprocessing isolation, and scoring."""
from itertools import combinations
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from facial_expression.data import CLASSES, ROOT, inferred_groups, load_data, make_splits, xy
from facial_expression import models
from facial_expression.train import evaluate


@pytest.fixture
def data():
    return load_data(ROOT / "data/features.csv")


@pytest.mark.parametrize("strategy,counts", [("image", [686, 147, 148]), ("inferred-sequence", [684, 147, 150])])
def test_splits_cover_every_image_once_and_are_repeatable(data, strategy, counts):
    parts = make_splits(data, strategy=strategy)
    repeated = make_splits(data, strategy=strategy)
    assert [len(part) for part in parts.values()] == counts
    assert set(pd.concat(parts.values())["image_name"]) == set(data["image_name"])
    for name, part in parts.items():
        pd.testing.assert_frame_equal(part, repeated[name])
        assert set(part["emotion"]) == set(CLASSES)
    for a, b in combinations(parts.values(), 2):
        assert set(a["image_name"]).isdisjoint(b["image_name"])
        if strategy == "inferred-sequence":
            assert set(inferred_groups(a)).isdisjoint(inferred_groups(b))


def test_identifiers_and_labels_never_enter_features(data):
    X, y = xy(data)
    assert X.shape == (981, 25)
    assert np.isfinite(X).all()
    assert set(y) == set(CLASSES)


def test_scaler_is_fitted_separately_in_every_cv_fold(monkeypatch, data):
    observed_sizes = []
    class ObservedScaler(StandardScaler):
        def fit(self, X, y=None, sample_weight=None):
            observed_sizes.append(len(X))
            return super().fit(X, y, sample_weight=sample_weight)
    monkeypatch.setattr(models, "StandardScaler", ObservedScaler)
    train = make_splits(data)["train"]
    search = models.make_search("svm", quick=True)
    search.fit(*xy(train))
    assert observed_sizes == [343, 343, 686]


def test_grouped_cv_keeps_inferred_frames_together(data):
    train = make_splits(data, strategy="inferred-sequence")["train"]
    X, y = xy(train)
    groups = inferred_groups(train)
    for fit, validation in models.make_search("svm", grouped=True).cv.split(X, y, groups):
        assert set(groups[fit]).isdisjoint(groups[validation])


def test_macro_f1_counts_minority_class_failure(data):
    class AlwaysSurprised:
        def predict(self, X):
            return np.repeat("surprise", len(X))
    scores, report, _ = evaluate(AlwaysSurprised(), data)
    expected_surprise_f1 = 2 * 249 / (981 + 249)
    assert scores["macro_f1"] == pytest.approx(expected_surprise_f1 / 7)
    assert scores["accuracy"] == pytest.approx(249 / 981)
    assert report["contempt"]["recall"] == 0


def test_duplicate_features_are_rejected(data, tmp_path):
    duplicate = data.copy()
    duplicate.iloc[1, 1:-1] = duplicate.iloc[0, 1:-1]
    path = tmp_path / "bad.csv"
    duplicate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate feature vectors"):
        load_data(path)


@pytest.mark.parametrize("directory", [ROOT / "results", ROOT / "results/inferred_sequence"])
def test_saved_scores_match_independent_predictions(directory):
    from sklearn.metrics import accuracy_score, f1_score
    metrics = json.loads((directory / "metrics.json").read_text())
    manifest = pd.read_csv(directory / "split_manifest.csv")
    expected = manifest.query("split == 'test'").set_index("image_name")["emotion"]
    for name, result in metrics["models"].items():
        pred = pd.read_csv(directory / f"predictions_{name}.csv")
        assert pred["image_name"].is_unique
        assert set(pred["image_name"]) == set(expected.index)
        assert (expected.loc[pred["image_name"]].to_numpy() == pred["true_label"].to_numpy()).all()
        scores = result["scores"]["test"]
        assert scores["macro_f1"] == pytest.approx(f1_score(pred["true_label"], pred["predicted_label"], labels=CLASSES, average="macro"))
        assert scores["accuracy"] == pytest.approx(accuracy_score(pred["true_label"], pred["predicted_label"]))


def test_mlp_seed_and_best_validation_checkpoint():
    import torch
    from sklearn.metrics import f1_score
    from facial_expression.neural import fit_network
    rng = np.random.default_rng(12)
    X = rng.normal(size=(140, 25)).astype(np.float32)
    y = np.tile(np.arange(7), 20)
    X_val = rng.normal(size=(35, 25)).astype(np.float32)
    y_val = np.tile(np.arange(7), 5)
    params = {"hidden_dim": 16, "dropout": .2, "lr": .01, "weight_decay": .001, "batch_size": 16}
    a, score, epoch, history = fit_network(X, y, X_val, y_val, params, 42, 8, 5)
    b, repeated_score, repeated_epoch, repeated_history = fit_network(X, y, X_val, y_val, params, 42, 8, 5)
    assert (score, epoch, history) == (repeated_score, repeated_epoch, repeated_history)
    with torch.no_grad():
        pred = a(torch.tensor(X_val)).argmax(1).numpy()
    assert f1_score(y_val, pred, labels=range(7), average="macro", zero_division=0) == score
    assert score == max(row["validation_macro_f1"] for row in history)
    assert all(torch.equal(a.state_dict()[key], b.state_dict()[key]) for key in a.state_dict())
