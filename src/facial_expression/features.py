"""Regenerate versioned geometric features from the archived MediaPipe landmarks.

Distances and horizontal symmetry offsets use the outer-eye span as their unit.
Aspect ratios are already dimensionless; angles are reported in degrees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 2
FEATURE_COLUMNS = [
    "left_eye_width", "left_eye_height", "right_eye_width", "right_eye_height",
    "left_eye_aspect_ratio", "right_eye_aspect_ratio",
    "left_eyebrow_eye_distance", "right_eyebrow_eye_distance", "inter_eyebrow_distance",
    "mouth_width", "mouth_height", "mouth_inner_width", "left_lip_height", "right_lip_height",
    "mouth_aspect_ratio", "nose_length", "nose_to_lip_distance", "jaw_width",
    "chin_to_nose_distance", "eye_line_angle", "mouth_line_angle",
    "eye_symmetry", "mouth_symmetry", "upper_eye_symmetry", "eyebrow_symmetry",
]
LANDMARK_COLUMNS = [f"x{i}" for i in range(468)] + [f"y{i}" for i in range(468)]
_SYMMETRY_PAIRS = [(33, 263), (61, 291), (159, 386), (105, 334)]


def _points(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (468, 2):
        raise ValueError("Expected a (468, 2) array of x/y MediaPipe landmarks")
    if not np.isfinite(points).all():
        raise ValueError("Landmarks must contain only finite numeric coordinates")
    return points


def _distance(points: np.ndarray, left: int, right: int) -> float:
    delta = points[left] - points[right]
    return float(np.hypot(delta[0], delta[1]))


def _angle(points: np.ndarray, left: int, right: int) -> float:
    delta = points[right] - points[left]
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def _positive(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 1e-12:
        raise ValueError(f"Degenerate landmarks: {name} must be positive")
    return value


def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """Return 25 features, invariant to translation and positive uniform scale.

    Coordinate axes and degree angles retain the original image orientation.
    This does not remove head rotation or correct unknown image aspect ratios.
    """
    points = _points(landmarks)
    span = _positive(_distance(points, 33, 263), "outer-eye span")
    left_width = _positive(_distance(points, 33, 133), "left eye width")
    left_height = _distance(points, 159, 145)
    right_width = _positive(_distance(points, 362, 263), "right eye width")
    right_height = _distance(points, 386, 374)
    mouth_width = _positive(_distance(points, 61, 291), "mouth width")
    mouth_height = _distance(points, 13, 14)
    symmetry = [
        abs((points[left, 0] - points[1, 0]) - (points[1, 0] - points[right, 0])) / span
        for left, right in _SYMMETRY_PAIRS
    ]
    features = np.asarray([
        left_width / span, left_height / span, right_width / span, right_height / span,
        left_height / left_width, right_height / right_width,
        _distance(points, 105, 159) / span, _distance(points, 334, 386) / span,
        _distance(points, 70, 300) / span,
        mouth_width / span, mouth_height / span, _distance(points, 78, 308) / span,
        _distance(points, 82, 87) / span, _distance(points, 312, 317) / span,
        mouth_height / mouth_width,
        _distance(points, 168, 1) / span, _distance(points, 1, 13) / span,
        _distance(points, 234, 454) / span, _distance(points, 152, 1) / span,
        _angle(points, 33, 263), _angle(points, 61, 291), *symmetry,
    ], dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Landmark geometry produced nonfinite feature values")
    return features


def regenerate_features(landmarks: pd.DataFrame) -> pd.DataFrame:
    """Generate the corrected schema, preserving image names, labels, and order."""
    required = ["image_name", *LANDMARK_COLUMNS, "emotion"]
    missing = set(required) - set(landmarks.columns)
    if missing:
        raise ValueError(f"Missing required landmark columns: {sorted(missing)}")
    if landmarks.empty or landmarks[required].isna().any().any():
        raise ValueError("Landmark table must be nonempty and contain no missing values")
    if not landmarks["image_name"].is_unique:
        raise ValueError("Landmark image names must be unique")
    coordinates = landmarks[LANDMARK_COLUMNS].to_numpy(dtype=np.float64)
    generated = []
    for name, values in zip(landmarks["image_name"], coordinates):
        try:
            generated.append(extract_features(values.reshape(2, 468).T))
        except ValueError as error:
            raise ValueError(f"Invalid landmarks for {name}: {error}") from error
    result = pd.DataFrame(generated, columns=FEATURE_COLUMNS)
    result.insert(0, "image_name", landmarks["image_name"].to_numpy())
    result["emotion"] = landmarks["emotion"].to_numpy()
    return result


def _legacy_features(landmarks: np.ndarray) -> np.ndarray:
    """Reproduce the archived formula solely to verify source-data provenance."""
    points = _points(landmarks)
    # np.linalg.norm and the historical epsilons intentionally reproduce v1.
    distance = lambda a, b: float(np.linalg.norm(points[a] - points[b]))
    span = distance(33, 263) + 1e-6
    eyes = [distance(33, 133), distance(159, 145), distance(362, 263), distance(386, 374)]
    eyes += [eyes[1] / (eyes[0] + 1e-6), eyes[3] / (eyes[2] + 1e-6)]
    brows = [distance(105, 159), distance(334, 386), distance(70, 300)]
    mouth = [distance(61, 291), distance(13, 14), distance(78, 308), distance(82, 87), distance(312, 317)]
    mouth += [mouth[1] / (mouth[0] + 1e-6)]
    nose_jaw = [distance(168, 1), distance(1, 13), distance(234, 454), distance(152, 1)]
    symmetry = [abs((points[left, 0] - points[1, 0]) - (points[1, 0] - points[right, 0]))
                for left, right in _SYMMETRY_PAIRS]
    return np.asarray([value / span for value in eyes + brows + mouth + nose_jaw]
                      + [_angle(points, 33, 362), _angle(points, 61, 291)] + symmetry)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def main(argv: list[str] | None = None) -> None:
    landmark_dir = ROOT / "archive/coursework/Preprocessing/CleanData2/LandmarkData"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmarks", type=Path, nargs="+", default=[
        landmark_dir / "landmarks_train.csv", landmark_dir / "landmarks_test.csv",
    ])
    parser.add_argument("--reference", type=Path, default=ROOT / "data/features_original.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/features.csv")
    parser.add_argument("--provenance", type=Path, default=ROOT / "data/provenance.json")
    args = parser.parse_args(argv)

    landmarks = pd.concat([pd.read_csv(path) for path in args.landmarks], ignore_index=True)
    reference = pd.read_csv(args.reference)
    if reference.empty or reference.isna().any().any() or not reference["image_name"].is_unique:
        raise ValueError("Reference features must be nonempty, complete, and have unique image names")
    if not landmarks["image_name"].is_unique:
        raise ValueError("Landmark image names must be unique across source files")
    if set(landmarks["image_name"]) != set(reference["image_name"]):
        raise ValueError("Landmarks and original features must have exactly the same image names")
    landmarks = landmarks.set_index("image_name").loc[reference["image_name"]].reset_index()
    if not np.array_equal(landmarks["emotion"], reference["emotion"]):
        raise ValueError("Landmark and original-feature labels do not match")
    expected_legacy_columns = [*FEATURE_COLUMNS[:-2], "eyebrow_symmetry", "facial_symmetry"]
    if list(reference.columns) != ["image_name", *expected_legacy_columns, "emotion"]:
        raise ValueError("Reference does not use the expected 25-feature coursework schema")
    coords = landmarks[LANDMARK_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 2, 468).transpose(0, 2, 1)
    reconstructed = np.asarray([_legacy_features(points) for points in coords])
    original_values = reference[expected_legacy_columns].to_numpy(dtype=np.float64)
    residual = np.abs(reconstructed - original_values)
    if not np.allclose(reconstructed, original_values, rtol=1e-9, atol=1e-9):
        raise ValueError("Archived landmarks do not reproduce the original features within 1e-9 tolerance")
    corrected = regenerate_features(landmarks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(args.output, index=False, lineterminator="\n")
    metadata = {
        "feature_schema_version": SCHEMA_VERSION,
        "rows": len(corrected),
        "feature_columns": FEATURE_COLUMNS,
        "class_counts": corrected["emotion"].value_counts().sort_index().to_dict(),
        "input_files": [{"path": _source_path(path), "sha256": _sha256(path)}
                        for path in [args.reference, *args.landmarks]],
        "output_file": {"path": _source_path(args.output), "sha256": _sha256(args.output)},
        "legacy_reconstruction": {
            "verified": True, "rtol": 1e-9, "atol": 1e-9,
            "maximum_absolute_error": float(residual.max()),
            "labels_match": True, "image_names_match": True,
        },
        "corrections": [
            "Keep eye and mouth aspect ratios dimensionless instead of dividing by eye span again.",
            "Normalize all four horizontal symmetry offsets by outer-eye span.",
            "Use outer corners 33 and 263 for eye_line_angle.",
            "Rename the last symmetry slots to upper_eye_symmetry and eyebrow_symmetry.",
            "Reject degenerate denominators instead of adding scale-dependent epsilon constants.",
        ],
        "grouping_limit": "Renamed images do not retain verified subject or sequence identifiers.",
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Regenerated {len(corrected)} rows with feature schema v{SCHEMA_VERSION}: {args.output}")
    print(f"Legacy provenance check passed; maximum absolute residual {residual.max():.3g}")


if __name__ == "__main__":
    main()
