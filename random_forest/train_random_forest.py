#!/usr/bin/env python3
"""
Train a Random-Forest classifier with optional group-wise down-sampling of
the majority class (“noise”), and compute principled RF threshold suggestions.

USER CONFIG (edit the constants below):
    CSV_CHOICE   – which processed feature file to train on
    NEG_PER_POS  – None, 1, 2, or 3   (noise rows kept per target row, per tape)
    N_TREES      – number of trees
"""

from pathlib import Path
import json
import joblib, numpy as np, pandas as pd, random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve
)
from split_utils import load_features, grouped_train_test

# ─────────── USER CONFIG ────────────────────────────────────────────
CSV_CHOICE = "actually_with_logits.csv"  # or "mfcc_with_labels.csv", etc.
NEG_PER_POS = 1  # None for raw; 1, 2, 3 for balanced subsets
N_TREES = 800
DATA_DIR = Path(
    "/Users/danie/repos/alpaca-segmentation/random_forest/data/"
    "spectral_feature_annotations/actually_with_logits"
)
MODEL_DIR = Path("models");
MODEL_DIR.mkdir(exist_ok=True)
METRIC_DIR = Path("metrics");
METRIC_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 0
# Threshold suggestion policy
REC_FLOOR = 0.93  # we care slightly more about recall
BETA = 1.25  # Fβ with β>1 weights recall a bit more
# ────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def downsample_by_group(df: pd.DataFrame, neg_per_pos: int | None) -> pd.DataFrame:
    """Return a frame where, within each tape, #noise ≈ neg_per_pos × #target."""
    if neg_per_pos is None:
        return df  # keep raw imbalance
    keep_rows = []
    for tape, sub in df.groupby("wave_file", sort=False):
        pos_idx = sub[sub.validated_label == "target"].index
        neg_idx = sub[sub.validated_label != "target"].index
        n_pos = len(pos_idx)
        n_keep_neg = min(len(neg_idx), neg_per_pos * n_pos)
        if n_keep_neg > 0:
            sampled_neg = np.random.choice(neg_idx, n_keep_neg, replace=False)
            keep_rows.extend(sampled_neg)
        keep_rows.extend(pos_idx)
    return df.loc[keep_rows]


def suggest_thresholds(y_true: pd.Series, proba: np.ndarray,
                       rec_floor: float = REC_FLOOR, beta: float = BETA):
    """
    Return two suggestions:
      - recall_first: highest t with recall>=rec_floor that minimises FP rate on negatives
      - fbeta_best  : t that maximises F_beta (beta>1 favours recall)
    Also return diagnostics (precision, recall, f2) at those points.
    """
    prec, rec, th = precision_recall_curve(y_true, proba)  # th has len-1
    # Align arrays: use indices over th, referencing prec[i], rec[i]
    # (prec/rec have one extra element at the end that corresponds to t beyond max(proba))
    n = len(th)
    # Compute confusion-derived f2 (FP rate on negatives) per threshold
    f2_list = []
    for i in range(n):
        t = th[i]
        pred_i = (proba >= t).astype(int)
        tn = ((y_true == 0) & (pred_i == 0)).sum()
        fp = ((y_true == 0) & (pred_i == 1)).sum()
        f2 = fp / max(1, (fp + tn))
        f2_list.append(f2)
    f2_arr = np.asarray(f2_list, dtype=float)

    # 1) Recall-first: among rec>=floor, pick smallest f2; tie-break by higher threshold
    mask = rec[:n] >= rec_floor
    if mask.any():
        cand_idx = np.where(mask)[0]
        # choose idx with minimal f2; if tie, prefer higher threshold (more conservative)
        best_i = int(cand_idx[np.lexsort((-th[cand_idx], f2_arr[cand_idx]))][0])
        reco = {
            "threshold": float(th[best_i]),
            "precision": float(prec[best_i]),
            "recall": float(rec[best_i]),
            "f2": float(f2_arr[best_i]),
            "policy": f"recall_first (rec≥{rec_floor:.2f}, min f2)"
        }
    else:
        # fallback: use 0.50 with its realised metrics
        t = 0.50
        pred_i = (proba >= t).astype(int)
        tp = ((y_true == 1) & (pred_i == 1)).sum()
        fp = ((y_true == 0) & (pred_i == 1)).sum()
        tn = ((y_true == 0) & (pred_i == 0)).sum()
        fn = ((y_true == 1) & (pred_i == 0)).sum()
        prec_t = tp / max(1, (tp + fp))
        rec_t = tp / max(1, (tp + fn))
        f2_t = fp / max(1, (fp + tn))
        reco = {
            "threshold": 0.50,
            "precision": float(prec_t),
            "recall": float(rec_t),
            "f2": float(f2_t),
            "policy": f"fallback_to_0.50 (no threshold had rec≥{rec_floor:.2f})"
        }

    # 2) F_beta optimum (over the same n thresholds)
    num = (1 + beta ** 2) * (prec[:n] * rec[:n])
    den = (beta ** 2) * prec[:n] + rec[:n]
    F = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    best_f_idx = int(np.nanargmax(F))
    fbeta = {
        "threshold": float(th[best_f_idx]),
        "precision": float(prec[best_f_idx]),
        "recall": float(rec[best_f_idx]),
        "fbeta": float(F[best_f_idx]),
        "beta": float(beta),
        "policy": f"F_beta (beta={beta:.2f})"
    }

    return reco, fbeta


# 1) load data + grouped split by tape
csv_path = DATA_DIR / CSV_CHOICE
print(f"▶ Loading {csv_path}")
df_full = load_features(csv_path)
X_tr, X_te, y_tr, y_te = grouped_train_test(df_full)

# 2) down-sample majority class inside train, then vectorise
train_frame = df_full.loc[X_tr.index]
train_frame = downsample_by_group(train_frame, NEG_PER_POS)

print(f"   After down-sampling: {train_frame.validated_label.value_counts()}")
print("▶ Vectorising features…")
X_train = train_frame.drop(columns=["validated_label", "wave_file", "selection"])
y_train = (train_frame.validated_label == "target").astype(int)
# test from grouped split (already feature-only)
X_test = X_te.reset_index(drop=True)
y_test = y_te.reset_index(drop=True)

# 3) fit RF
print("▶ Fitting Random Forest…")
rf = RandomForestClassifier(
    n_estimators=N_TREES,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_SEED
)
rf.fit(X_train, y_train)

# 4) evaluate at default 0.50 (for continuity with previous reports)
prob = rf.predict_proba(X_test)[:, 1]
pred = (prob >= 0.50).astype(int)

auroc = roc_auc_score(y_test, prob)
aupr = average_precision_score(y_test, prob)
report = classification_report(y_test, pred, digits=3)
cm = confusion_matrix(y_test, pred)

# 5) threshold suggestions from PR sweep
reco_suggest, fbeta_suggest = suggest_thresholds(y_test, prob, REC_FLOOR, BETA)

stem = csv_path.stem + f"_neg{NEG_PER_POS or 'raw'}"
model_path = MODEL_DIR / f"rf_{stem}.pkl"
metric_path = METRIC_DIR / f"rf_{stem}_report.txt"
thjson_path = METRIC_DIR / f"rf_{stem}_thresholds.json"

joblib.dump(rf, model_path)

# Write metrics report (text)
with metric_path.open("w") as fh:
    fh.write(f"Random Forest on {CSV_CHOICE}  (neg_per_pos={NEG_PER_POS})\n")
    fh.write(report + "\n")
    fh.write(f"Confusion:\n{cm}\n")
    fh.write(f"ROC-AUC : {auroc:0.4f}   PR-AUC : {aupr:0.4f}\n\n")
    fh.write("Threshold suggestions:\n")
    fh.write(f"  - recall_first: t={reco_suggest['threshold']:.3f}  "
             f"(precision={reco_suggest['precision']:.3f}, "
             f"recall={reco_suggest['recall']:.3f}, "
             f"f2={reco_suggest['f2']:.3f}; policy={reco_suggest['policy']})\n")
    if "fbeta" in fbeta_suggest:
        fh.write(f"  - F_beta      : t={fbeta_suggest['threshold']:.3f}  "
                 f"(precision={fbeta_suggest['precision']:.3f}, "
                 f"recall={fbeta_suggest['recall']:.3f}, "
                 f"F_beta={fbeta_suggest['fbeta']:.3f}; beta={fbeta_suggest['beta']:.2f})\n")
    else:
        fh.write(f"  - F_beta      : t={fbeta_suggest['threshold']:.3f}\n")

# Also write a tiny JSON sidecar for code to consume later (optional but handy)
th_payload = {
    "csv_choice": CSV_CHOICE,
    "neg_per_pos": NEG_PER_POS,
    "recall_floor": REC_FLOOR,
    "beta": BETA,
    "suggestions": {
        "recall_first": reco_suggest,
        "f_beta": fbeta_suggest
    }
}
thjson_path.write_text(json.dumps(th_payload, indent=2))

# Console summary
print(f"\n────────  SUMMARY  (neg_per_pos={NEG_PER_POS}) ────────")
print(report)
print(f"ROC-AUC : {auroc:0.4f}    PR-AUC : {aupr:0.4f}")
print("Confusion matrix [TN FP; FN TP]:")
print(cm)
print("─ Threshold suggestions ─")
print(f"  recall_first → t={reco_suggest['threshold']:.3f}  "
      f"(P={reco_suggest['precision']:.3f}, R={reco_suggest['recall']:.3f}, f2={reco_suggest['f2']:.3f})")
print(f"  F_beta(β={BETA:.2f}) → t={fbeta_suggest['threshold']:.3f}  "
      f"(P={fbeta_suggest['precision']:.3f}, R={fbeta_suggest['recall']:.3f}, Fβ={fbeta_suggest['fbeta']:.3f})")
print("───────────────────────────────────────────────────────")
print("✓ Model saved   →", model_path)
print("✓ Metrics saved →", metric_path)
print("✓ Thresholds    →", thjson_path)
