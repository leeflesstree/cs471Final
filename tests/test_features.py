"""Geometry contracts and regeneration against the actual archived dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from facial_expression.features import FEATURE_COLUMNS, ROOT, extract_features, main, regenerate_features


@pytest.fixture
def landmark_table():
    directory = ROOT / "archive/coursework/Preprocessing/CleanData2/LandmarkData"
    return pd.concat([pd.read_csv(directory / f"landmarks_{split}.csv")
                      for split in ["train", "test"]], ignore_index=True)


@pytest.fixture
def points(landmark_table):
    row = landmark_table.iloc[0]
    return np.asarray([[row[f"x{i}"], row[f"y{i}"]] for i in range(468)])


def test_translation_and_positive_scale_invariance(points):
    expected = extract_features(points)
    for scale, translation in [(0.5, [0.2, -0.3]), (2.0, [-0.5, 0.1]), (10.0, [8, -4])]:
        np.testing.assert_allclose(extract_features(points * scale + translation), expected,
                                   rtol=1e-10, atol=1e-10)


def test_dimensionless_ratios_and_outer_eye_angle(points):
    values = dict(zip(FEATURE_COLUMNS, extract_features(points)))
    assert len(values) == 25
    assert values["left_eye_aspect_ratio"] == pytest.approx(values["left_eye_height"] / values["left_eye_width"])
    assert values["right_eye_aspect_ratio"] == pytest.approx(values["right_eye_height"] / values["right_eye_width"])
    assert values["mouth_aspect_ratio"] == pytest.approx(values["mouth_height"] / values["mouth_width"])
    expected_angle = np.degrees(np.arctan2(*(points[263] - points[33])[::-1]))
    assert values["eye_line_angle"] == pytest.approx(expected_angle)


def test_invalid_geometry_fails_clearly(points):
    with pytest.raises(ValueError, match="468, 2"):
        extract_features(points[:-1])
    invalid = points.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        extract_features(invalid)
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        extract_features(invalid)
    with pytest.raises(ValueError, match="outer-eye span"):
        extract_features(np.zeros((468, 2)))
    invalid = points.copy()
    invalid[133] = invalid[33]
    with pytest.raises(ValueError, match="left eye width"):
        extract_features(invalid)
    invalid = points.copy()
    invalid[291] = invalid[61]
    with pytest.raises(ValueError, match="mouth width"):
        extract_features(invalid)


def test_regeneration_preserves_real_dataset_identity(landmark_table):
    generated = regenerate_features(landmark_table)
    assert list(generated.columns) == ["image_name", *FEATURE_COLUMNS, "emotion"]
    assert len(generated) == 981
    assert generated["image_name"].tolist() == landmark_table["image_name"].tolist()
    assert generated["emotion"].tolist() == landmark_table["emotion"].tolist()
    assert np.isfinite(generated[FEATURE_COLUMNS].to_numpy()).all()
    original = pd.read_csv(ROOT / "data/features_original.csv")
    assert set(generated["image_name"]) == set(original["image_name"])
    with pytest.raises(ValueError, match="unique"):
        regenerate_features(pd.concat([landmark_table, landmark_table.iloc[:1]]))


def test_cli_verifies_legacy_provenance_and_reference_order(tmp_path):
    import json

    output = tmp_path / "features.csv"
    provenance = tmp_path / "provenance.json"
    main(["--output", str(output), "--provenance", str(provenance)])
    generated = pd.read_csv(output)
    original = pd.read_csv(ROOT / "data/features_original.csv")
    assert generated["image_name"].tolist() == original["image_name"].tolist()
    assert generated["emotion"].tolist() == original["emotion"].tolist()
    metadata = json.loads(provenance.read_text())
    assert metadata["feature_schema_version"] == 2
    assert metadata["rows"] == 981
    assert metadata["legacy_reconstruction"]["verified"]
    assert metadata["legacy_reconstruction"]["maximum_absolute_error"] < 1e-9
    assert len(metadata["input_files"]) == 3
    assert all(len(source["sha256"]) == 64 for source in metadata["input_files"])
