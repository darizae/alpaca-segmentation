#!/usr/bin/env python3
"""
Train a Random-Forest classifier with optional group-wise down-sampling of
the majority class (“noise”).

USER CONFIG (edit the constants below):
    CSV_CHOICE   – which processed feature file to train on
    NEG_PER_POS  – None, 1, 2, or 3   (noise rows kept per target row, per tape)
    N_TREES      – number of trees
"""

from pathlib import Path
import joblib, numpy as np, pandas as pd, random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)
from split_utils import load_features, grouped_train_test

# ─────────── USER CONFIG ────────────────────────────────────────────
CSV_CHOICE = "spectral_robust_py.csv"  # or "mfcc_with_labels.csv" …
NEG_PER_POS = 2  # None for raw; 1, 2, 3 for balanced subsets
N_TREES = 800
DATA_DIR = Path(
    "/Users/danie/repos/alpaca-segmentation/random_forest/data/"
    "spectral_feature_annotations/py"
)
MODEL_DIR = Path("models");
MODEL_DIR.mkdir(exist_ok=True)
METRIC_DIR = Path("metrics");
METRIC_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 0
# ────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED);
np.random.seed(RANDOM_SEED)


def downsample_by_group(df: pd.DataFrame, neg_per_pos: int | None) -> pd.DataFrame:
    """Return a frame where, within each tape, #noise ≈ neg_per_pos × #target."""
    if neg_per_pos is None:
        return df  # keep raw imbalance
    keep_rows = []
    for tape, sub in df.groupby("wave_file"):
        pos_idx = sub[sub.validated_label == "target"].index
        neg_idx = sub[sub.validated_label != "target"].index
        n_pos = len(pos_idx)
        n_keep_neg = min(len(neg_idx), neg_per_pos * n_pos)
        sampled_neg = np.random.choice(neg_idx, n_keep_neg, replace=False)
        keep_rows.extend(pos_idx)
        keep_rows.extend(sampled_neg)
    return df.loc[keep_rows]


# 1. load data + split by tape
csv_path = DATA_DIR / CSV_CHOICE
print(f"▶ Loading {csv_path}")
df_full = load_features(csv_path)
X_tr, X_te, y_tr, y_te = grouped_train_test(df_full)

# 2. merge back to a frame so we can down-sample, then split again
train_frame = df_full.loc[X_tr.index]  # exact rows selected
train_frame = downsample_by_group(train_frame, NEG_PER_POS)

print(f"   After down-sampling: {train_frame.validated_label.value_counts()}")
print("▶ Vectorising features…")
X_train = train_frame.drop(columns=["validated_label", "wave_file", "selection"])
y_train = (train_frame.validated_label == "target").astype(int)

X_test = X_te.reset_index(drop=True)  # already vectorised
y_test = y_te.reset_index(drop=True)

# 3. fit RF
print("▶ Fitting Random Forest…")
rf = RandomForestClassifier(
    n_estimators=N_TREES,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_SEED)
rf.fit(X_train, y_train)

# 4. evaluate
prob = rf.predict_proba(X_test)[:, 1]
pred = rf.predict(X_test)

auroc = roc_auc_score(y_test, prob)
aupr = average_precision_score(y_test, prob)
report = classification_report(y_test, pred, digits=3)
cm = confusion_matrix(y_test, pred)

stem = csv_path.stem + f"_neg{NEG_PER_POS or 'raw'}"
model_path = MODEL_DIR / f"rf_{stem}.pkl"
metric_path = METRIC_DIR / f"rf_{stem}_report.txt"
joblib.dump(rf, model_path)

with metric_path.open("w") as fh:
    fh.write(f"Random Forest on {CSV_CHOICE}  (neg_per_pos={NEG_PER_POS})\n")
    fh.write(report + "\n")
    fh.write(f"Confusion:\n{cm}\n")
    fh.write(f"ROC-AUC : {auroc:0.4f}   PR-AUC : {aupr:0.4f}\n")

print(f"\n────────  SUMMARY  (neg_per_pos={NEG_PER_POS}) ────────")
print(report)
print(f"ROC-AUC : {auroc:0.4f}    PR-AUC : {aupr:0.4f}")
print("Confusion matrix [TN FP; FN TP]:")
print(cm)
print("───────────────────────────────────────────────────────")
print("✓ Model saved →", model_path)
print("✓ Metrics saved →", metric_path)
