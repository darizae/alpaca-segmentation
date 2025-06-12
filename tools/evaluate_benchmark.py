#!/usr/bin/env python3
"""Evaluate alpaca-hum segmentation benchmark runs.

Metrics per (model, variant)
---------------------------
* **precision**  (upper-bound – GT omissions count as FP)
* **recall**     (lower-bound – misses are real)
* **F1**         (harmonic mean)
* **mean |Δstart| & |Δend|** in **ms** on matched calls
* **Pearson / Spearman** correlation of per-tape call counts
* **Q1-only recall**

Differences to first draft
-------------------------
Ground-truth and prediction index files use different schemas:

* **GT** (`corpus_index.json`)
    ```json
    {
        "hum_path": "…/UNKN_20250205.wav_0_3594.wav_1263…Q2.wav",
        "clip_path": "labelled_recordings_new/UNKN_20250205.wav_0_3594.wav",
        "raw_start_s": 1263.318,
        "raw_end_s":   1263.686,
        "quality": 2,
        …
    }
    ```
* **Prediction** (`evaluation/index.json`)
    ```json
    {
        "pred_path": "evaluation/…result.txt",
        "tape": "UNKN_20250205.wav",
        "start_s": 1263.29,
        "end_s":   1263.68,
        …
    }
    ```

The refactor introduces separate loaders that normalise both into the common
columns `[tape, beg, end, quality]`.

Run example (from repo root):
```
python tools/evaluate_benchmark.py \
  --gt data/benchmark_corpus_v1/corpus_index.json \
  --runs BENCHMARK/runs \
  --iou 0.40 \
  --out metrics.csv
```

Dependencies: `pandas≥1.3`, `numpy`, `scipy`, `tqdm`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

###############################################################################
# Helpers
###############################################################################

def tape_from_clip(clip_path: str) -> str:
    """Derive raw tape name (<tape>.wav) from a clipped filename."""
    name = Path(clip_path).name  # e.g. UNKN_…obs.wav_0_3594.wav
    if ".wav_" in name:
        return name.split(".wav_")[0] + ".wav"
    return name  # fallback – already looks like a tape


def read_gt_index(idx_path: Path) -> pd.DataFrame:
    """Return DF with cols [tape, beg, end, quality]."""
    idx = json.loads(idx_path.read_text())
    rows = []
    for e in idx["entries"]:
        rows.append(
            {
                "tape": tape_from_clip(e["clip_path"]),
                "beg": float(e.get("raw_start_s") or e["clip_start_s"]),
                "end": float(e.get("raw_end_s") or e["clip_end_s"]),
                "quality": int(e.get("quality", 1)),
            }
        )
    return pd.DataFrame(rows)


def read_pred_index(idx_path: Path) -> pd.DataFrame:
    idx = json.loads(idx_path.read_text())
    rows = [
        {"tape": e["tape"], "beg": float(e["start_s"]), "end": float(e["end_s"])}
        for e in idx["entries"]
    ]
    return pd.DataFrame(rows)


def interval_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def match_predictions(gt: pd.DataFrame, pr: pd.DataFrame, thr: float):
    """Greedy one-to-one match → returns masks + mapping list."""
    gt = gt.sort_values(["tape", "beg"], ignore_index=True)
    pr = pr.sort_values(["tape", "beg"], ignore_index=True)

    m_gt = np.zeros(len(gt), dtype=bool)
    m_pr = np.zeros(len(pr), dtype=bool)
    pairs: List[Tuple[int, int]] = []

    gt_grp = gt.groupby("tape").indices
    pr_grp = pr.groupby("tape").indices

    for tape, g_idx in gt_grp.items():
        if tape not in pr_grp:
            continue
        p_idx = list(pr_grp[tape])
        for gi in g_idx:
            best_iou, best_pi = 0.0, None
            g_int = (gt.at[gi, "beg"], gt.at[gi, "end"])
            for pi in p_idx:
                if m_pr[pi]:
                    continue
                iou = interval_iou(g_int, (pr.at[pi, "beg"], pr.at[pi, "end"]))
                if iou >= thr and iou > best_iou:
                    best_iou, best_pi = iou, pi
            if best_pi is not None:
                m_gt[gi] = True
                m_pr[best_pi] = True
                pairs.append((gi, best_pi))
    return m_gt, m_pr, pairs, gt, pr


def boundary_errors(pairs, gt, pr):
    if not pairs:
        return np.nan, np.nan
    ds, de = [], []
    for gi, pi in pairs:
        ds.append(abs(gt.at[gi, "beg"] - pr.at[pi, "beg"]) * 1000)
        de.append(abs(gt.at[gi, "end"] - pr.at[pi, "end"]) * 1000)
    return np.mean(ds), np.mean(de)


def per_tape_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby("tape").size()

###############################################################################
# Evaluation of single variant
###############################################################################

def eval_variant(gt_df: pd.DataFrame, pred_idx: Path, iou: float) -> Dict:
    pr_df = read_pred_index(pred_idx)
    m_gt, m_pr, pairs, gt_s, pr_s = match_predictions(gt_df, pr_df, iou)

    tp = int(m_gt.sum())
    fn = int((~m_gt).sum())
    fp = int((~m_pr).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    dstart_ms, dend_ms = boundary_errors(pairs, gt_s, pr_s)

    ct_gt = per_tape_counts(gt_df)
    ct_pr = per_tape_counts(pr_df).reindex(ct_gt.index, fill_value=0)
    pear = pearsonr(ct_gt, ct_pr)[0] if len(ct_gt) > 1 else np.nan
    spear = spearmanr(ct_gt, ct_pr)[0] if len(ct_gt) > 1 else np.nan

    # Q1-only recall
    gt_q1 = gt_df[gt_df.quality == 1]
    pairs_q1 = [p for p in pairs if gt_s.at[p[0], "quality"] == 1]
    rec_q1 = len(pairs_q1) / len(gt_q1) if len(gt_q1) else np.nan

    return {
        "n_gt": len(gt_df),
        "n_pred": len(pr_df),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_dstart_ms": dstart_ms,
        "mean_dend_ms": dend_ms,
        "pearson_calls": pear,
        "spearman_calls": spear,
        "recall_q1": rec_q1,
    }

###############################################################################
# CLI driver
###############################################################################

def tag_from(run_root: Path) -> Tuple[str, str]:
    return run_root.parts[-2], run_root.parts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Path to corpus_index.json")
    ap.add_argument("--runs", required=True, help="Folder with benchmark runs")
    ap.add_argument("--iou", type=float, default=0.40, help="IoU threshold")
    ap.add_argument("--out", default="metrics.csv", help="Output CSV file")
    args = ap.parse_args()

    gt_df = read_gt_index(Path(args.gt))

    run_roots = [p for p in Path(args.runs).glob("*/*") if (p / "evaluation/index.json").exists()]
    if not run_roots:
        raise SystemExit("❌ no run results found")

    records: List[Dict] = []
    for run in tqdm(run_roots, desc="variants"):
        model, variant = tag_from(run)
        rec = eval_variant(gt_df, run / "evaluation/index.json", args.iou)
        rec.update({"model": model, "variant": variant})
        records.append(rec)

    pd.DataFrame(records).to_csv(args.out, index=False)
    print(f"✅ metrics written → {args.out}")


if __name__ == "__main__":
    main()
