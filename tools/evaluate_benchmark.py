#!/usr/bin/env python3
"""evaluate_benchmark.py  ‒  event‑level metrics for alpaca‐hum segmentation

❗ **Update (mid‑point matcher)**
Instead of IoU ≥ τ we now count a prediction as a hit **if the mid‑point of the
predicted interval lies inside any ground‑truth interval** (one‑to‑one greedy
assignment).  All downstream stats remain unchanged, so existing notebooks
still digest the CSV.

Run example:

```bash
python tools/evaluate_benchmark.py \
  --gt data/benchmark_corpus_v1/corpus_index.json \
  --runs BENCHMARK/runs \
  --out metrics.csv
```

Dependencies: pandas, numpy, scipy, tqdm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

###############################################################################
# Helpers – loaders
###############################################################################


def tape_from_clip(clip_path: str) -> str:
    name = Path(clip_path).name
    if ".wav_" in name:
        return name.split(".wav_")[0] + ".wav"
    return name


def read_gt_index(p: Path) -> pd.DataFrame:
    j = json.loads(p.read_text())
    rows = []
    for e in j["entries"]:
        rows.append(
            {
                "tape": tape_from_clip(e["clip_path"]),
                "beg": float(e.get("raw_start_s") or e["clip_start_s"]),
                "end": float(e.get("raw_end_s") or e["clip_end_s"]),
                "quality": int(e.get("quality", 1)),
            }
        )
    return pd.DataFrame(rows)


def read_pred_index(p: Path) -> pd.DataFrame:
    j = json.loads(p.read_text())
    return pd.DataFrame(
        {k: [e[k] for e in j["entries"]] for k in ("tape", "start_s", "end_s")}
    ).rename(columns={"start_s": "beg", "end_s": "end"})

###############################################################################
# Matching – mid‑point inside GT
###############################################################################

def match_midpoint(gt: pd.DataFrame, pr: pd.DataFrame):
    """Greedy one‑to‑one assignment using mid‑point rule.

    Returns
    -------
    m_gt, m_pr : boolean masks
    pairs      : list of (gt_idx, pr_idx)
    gt_sort, pr_sort : sorted copies for downstream lookup
    """
    gt_sort = gt.sort_values(["tape", "beg"], ignore_index=True)
    pr_sort = pr.sort_values(["tape", "beg"], ignore_index=True)

    m_gt = np.zeros(len(gt_sort), dtype=bool)
    m_pr = np.zeros(len(pr_sort), dtype=bool)
    pairs: List[Tuple[int, int]] = []

    gt_grp = gt_sort.groupby("tape").indices
    pr_grp = pr_sort.groupby("tape").indices

    for tape, g_idx in gt_grp.items():
        if tape not in pr_grp:
            continue
        preds = pr_grp[tape]
        for gi in g_idx:
            beg_g, end_g = gt_sort.at[gi, "beg"], gt_sort.at[gi, "end"]
            for pi in preds:
                if m_pr[pi]:
                    continue
                mid_p = 0.5 * (pr_sort.at[pi, "beg"] + pr_sort.at[pi, "end"])
                if beg_g <= mid_p <= end_g:
                    m_gt[gi] = True
                    m_pr[pi] = True
                    pairs.append((gi, pi))
                    break  # move to next GT
    return m_gt, m_pr, pairs, gt_sort, pr_sort

###############################################################################
# Metrics for one variant
###############################################################################

def per_tape_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby("tape").size()


def boundary_errors(pairs, gt, pr):
    if not pairs:
        return np.nan, np.nan
    ds = [abs(gt.at[gi, "beg"] - pr.at[pi, "beg"]) * 1000 for gi, pi in pairs]
    de = [abs(gt.at[gi, "end"] - pr.at[pi, "end"]) * 1000 for gi, pi in pairs]
    return np.mean(ds), np.mean(de)


def eval_variant(gt_df: pd.DataFrame, pred_idx: Path) -> Dict:
    pr_df = read_pred_index(pred_idx)
    m_gt, m_pr, pairs, gt_s, pr_s = match_midpoint(gt_df, pr_df)

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

    # Quality‑specific recall & F1
    metrics_q = {}
    for q in range(1, 5):
        gt_q = gt_s[gt_s.quality == q]
        if gt_q.empty:
            metrics_q[f"recall_q{q}"] = np.nan
            metrics_q[f"f1_q{q}"] = np.nan
            continue
        pairs_q = [p for p in pairs if gt_s.at[p[0], "quality"] == q]
        tp_q = len(pairs_q)
        fn_q = len(gt_q) - tp_q
        rec_q = tp_q / len(gt_q)
        # precision_q: restrict preds to those matched to this quality
        preds_q = len(pairs_q)
        fp_q = preds_q - tp_q  # zero by construction
        prec_q = tp_q / (tp_q + fp_q) if (tp_q + fp_q) else np.nan
        f1_q = 2 * prec_q * rec_q / (prec_q + rec_q) if (prec_q + rec_q) else np.nan
        metrics_q[f"recall_q{q}"] = rec_q
        metrics_q[f"f1_q{q}"] = f1_q

    out = {
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
    }
    out.update(metrics_q)
    return out

###############################################################################
# CLI driver
###############################################################################

def tag_from(run_root: Path) -> Tuple[str, str]:
    return run_root.parts[-2], run_root.parts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Path to corpus_index.json")
    ap.add_argument("--runs", required=True, help="Folder with benchmark runs")
    ap.add_argument("--out", default="metrics.csv", help="Output CSV file")
    args = ap.parse_args()

    gt_df = read_gt_index(Path(args.gt))

    run_roots = [p for p in Path(args.runs).glob("*/*") if (p / "evaluation/index.json").exists()]
    if not run_roots:
        raise SystemExit("❌ no run results found")

    recs: List[Dict] = []
    for run in tqdm(run_roots, desc="variants"):
        model, variant = tag_from(run)
        res = eval_variant(gt_df, run / "evaluation/index.json")
        res.update({"model": model, "variant": variant})
        recs.append(res)

    pd.DataFrame(recs).to_csv(args.out, index=False)
    print(f"✅ metrics written → {args.out}")


if __name__ == "__main__":
    main()
