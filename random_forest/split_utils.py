#!/usr/bin/env python3
"""
split_utils.py
Shared helpers for loading a labelled feature CSV and returning a
group-preserving train/test split.

• load_features(path)              → pandas.DataFrame
• grouped_train_test(df, test_pct) → X_train, X_test, y_train, y_test

Author: Daniel’s AI coding-sidekick
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

__all__ = ["load_features", "grouped_train_test"]

def load_features(csv_path: str | Path) -> pd.DataFrame:
    """Read a processed feature file and sanity-check expected columns."""
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature file not found: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"validated_label", "wave_file", "selection"}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(f"{csv_path} lacks required columns: {missing}")

    return df


def grouped_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified split that keeps all windows from the same tape (wave_file)
    in either train *or* test.
    """
    X = df.drop(columns=["validated_label", "wave_file", "selection"])
    y = (df["validated_label"] == "target").astype(int)
    groups = df["wave_file"]

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )
