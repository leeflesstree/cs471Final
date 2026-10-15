"""Validate the dataset and create a single, explicit image-level split."""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.columns[0] != "image_name" or df.columns[-1] != "emotion" or len(df.columns) != 27:
        raise ValueError("Expected image_name, 25 feature columns, and emotion, in that order")
    if df.empty or df.isna().any().any():
        raise ValueError("Dataset must be nonempty and contain no missing values")
    if not df["image_name"].is_unique:
        raise ValueError("Image names must uniquely identify rows")
    if set(df["emotion"]) != set(CLASSES):
        raise ValueError(f"Expected all seven classes: {CLASSES}")
    features = df.iloc[:, 1:-1].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Features must be finite numeric values")
    if features.duplicated().any():
        raise ValueError("Duplicate feature vectors require investigation before splitting")
    df.iloc[:, 1:-1] = features
    return df


def inferred_groups(df: pd.DataFrame) -> np.ndarray:
    """Hypothesis: every three consecutive indices within an emotion share a sequence.

    These are NOT verified original CK+ sequence IDs and do not identify people.
    """
    groups = []
    for name, emotion in zip(df["image_name"], df["emotion"]):
        match = re.fullmatch(r"Ck\+_([a-z]+)_(\d+)\.png", name)
        if match is None or match.group(1) != emotion:
            raise ValueError(f"Cannot infer a frame group from {name!r}")
        groups.append(f"{emotion}_{int(match.group(2)) // 3}")
    return np.asarray(groups)


def make_splits(df: pd.DataFrame, seed: int = 42, strategy: str = "image") -> dict[str, pd.DataFrame]:
    """70/15/15 stratified random split; does NOT separate subjects or sequences."""
    if strategy == "image":
        units = df
    elif strategy == "inferred-sequence":
        units = pd.DataFrame({"group": inferred_groups(df), "emotion": df["emotion"].to_numpy()})
        if not units.groupby("group").size().eq(3).all():
            raise ValueError("Inferred-sequence sensitivity requires exactly three rows per group")
        units = units.drop_duplicates("group")
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    train, rest = train_test_split(
        units, test_size=0.3, stratify=units["emotion"], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest["emotion"], random_state=seed
    )
    splits = {"train": train, "validation": val, "test": test}
    if strategy == "inferred-sequence":
        groups = inferred_groups(df)
        splits = {name: df.loc[np.isin(groups, part["group"])] for name, part in splits.items()}
    return splits


def xy(df: pd.DataFrame):
    return df.iloc[:, 1:-1].to_numpy(dtype=np.float64), df["emotion"].to_numpy()
