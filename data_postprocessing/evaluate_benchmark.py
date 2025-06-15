#!/usr/bin/env python3
"""
Evaluate alpaca-hum segmentation benchmark runs.

Adds *per-tape* metrics:
---------------------------------
Besides the existing aggregate metrics the script now
writes a second CSV (``--per-tape-out``; default
``metrics_per_tape.csv``) that contains one row for every
(model, variant, tape) triple.

Usage example (from repo root)
------------------------------
python data_postprocessing/evaluate_benchmark.py               \
    --gt data/benchmark_corpus_v1/corpus_index.json            \
    --runs BENCHMARK/runs                                      \
    --iou 0.40                                                 \
    --out metrics.csv                                          \
    --per-tape-out metrics_per_tape.csv
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
# Helpers (unchanged)
###############################################################################

def tape_from_clip(clip_path: str) -> str:
    name = Path(clip_path).name
    if ".wav_" in name:
        return name.split(".wav_")[0] + ".wav"
    return name


TAPE_RX = re.compile(r"(.+?\.wav)", re.IGNORECASE)


def canonical_tape(name: str) -> str:
    """
    Return '<whatever>.wav' (first occurrence, case-insensitive).
    Falls back to the untouched string if no match is found.
    """
    m = TAPE_RX.search(name)
    return m.group(1) if m else name


def read_gt_index(idx_path: Path) -> pd.DataFrame:
    idx = json.loads(idx_path.read_text())
    rows = []
    for e in idx["entries"]:
        rows.append(
            {
                "tape": canonical_tape(Path(e["clip_path"]).name),
                "beg": float(e.get("raw_start_s") or e["clip_start_s"]),
                "end": float(e.get("raw_end_s") or e["clip_end_s"]),
                "quality": int(e.get("quality", 1)),
            }
        )
    return pd.DataFrame(rows)


def read_pred_index(idx_path: Path) -> pd.DataFrame:
    idx = json.loads(idx_path.read_text())
    rows = [
        {
            "tape": canonical_tape(e["tape"]),
            "beg": float(e["start_s"]),
            "end": float(e["end_s"]),
        }
        for e in idx["entries"]
    ]
    return pd.DataFrame(rows)


def interval_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def match_predictions(gt: pd.DataFrame, pr: pd.DataFrame, thr: float):
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
# Metric calculators
###############################################################################

def _base_metrics(m_gt, m_pr, pairs, gt_s, pr_s, gt_df, pr_df):
    """Shared block used for both global and per-tape evaluation."""
    tp = int(m_gt.sum())
    fn = int((~m_gt).sum())
    fp = int((~m_pr).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    dstart_ms, dend_ms = boundary_errors(pairs, gt_s, pr_s)

    metrics_q = {}
    for q in (1, 2, 3):
        gt_q = gt_df[gt_df.quality == q]
        pairs_q = [p for p in pairs if gt_s.at[p[0], "quality"] == q]
        tp_q = len(pairs_q)
        fn_q = len(gt_q) - tp_q
        fp_q = sum(gt_s.at[p[0], "quality"] != q for p in pairs)
        rec_q = tp_q / len(gt_q) if len(gt_q) else np.nan
        prec_q = tp_q / (tp_q + fp_q) if (tp_q + fp_q) else np.nan
        f1_q = 2 * prec_q * rec_q / (prec_q + rec_q) if (prec_q + rec_q) else np.nan
        metrics_q.update({f"recall_q{q}": rec_q, f"f1_q{q}": f1_q})

    return dict(
        n_gt=len(gt_df),
        n_pred=len(pr_df),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=prec,
        recall=rec,
        f1=f1,
        mean_dstart_ms=dstart_ms,
        mean_dend_ms=dend_ms,
        **metrics_q,
    )


def eval_variant(gt_df: pd.DataFrame, pred_idx: Path, iou: float) -> Dict:
    """Global (all-tapes-pooled) metrics – unchanged from your original."""
    pr_df = read_pred_index(pred_idx)
    m_gt, m_pr, pairs, gt_s, pr_s = match_predictions(gt_df, pr_df, iou)

    base = _base_metrics(m_gt, m_pr, pairs, gt_s, pr_s, gt_df, pr_df)

    # Per-tape call-rate correlations stay global
    ct_gt = per_tape_counts(gt_df)
    ct_pr = per_tape_counts(pr_df).reindex(ct_gt.index, fill_value=0)
    base["pearson_calls"] = pearsonr(ct_gt, ct_pr)[0] if len(ct_gt) > 1 else np.nan
    base["spearman_calls"] = spearmanr(ct_gt, ct_pr)[0] if len(ct_gt) > 1 else np.nan
    return base


def eval_variant_per_tape(gt_df: pd.DataFrame, pred_idx: Path, iou: float) -> List[Dict]:
    """Return one record per tape for the given variant."""
    pr_df = read_pred_index(pred_idx)
    all_tapes = sorted(set(gt_df.tape.unique()).union(pr_df.tape.unique()))
    records = []

    for tape in all_tapes:
        gt_t = gt_df[gt_df.tape == tape]
        pr_t = pr_df[pr_df.tape == tape]

        m_gt, m_pr, pairs, gt_s, pr_s = match_predictions(gt_t, pr_t, iou)
        rec = _base_metrics(m_gt, m_pr, pairs, gt_s, pr_s, gt_t, pr_t)
        rec["tape"] = tape
        records.append(rec)

    return records


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
    ap.add_argument("--out", default="metrics.csv", help="Global metrics CSV")
    ap.add_argument(
        "--per-tape-out",
        default="metrics_per_tape.csv",
        help="Per-tape metrics CSV",
    )
    args = ap.parse_args()

    gt_df = read_gt_index(Path(args.gt))

    run_roots = [
        p for p in Path(args.runs).glob("*/*") if (p / "evaluation/index.json").exists()
    ]
    if not run_roots:
        raise SystemExit("❌ no run results found")

    # global + per-tape accumulators
    glob_records: List[Dict] = []
    tape_records: List[Dict] = []

    for run in tqdm(run_roots, desc="variants"):
        model, variant = tag_from(run)
        pred_idx = run / "evaluation/index.json"

        # overall
        rec = eval_variant(gt_df, pred_idx, args.iou)
        rec.update(model=model, variant=variant)
        glob_records.append(rec)

        # per-tape
        for r in eval_variant_per_tape(gt_df, pred_idx, args.iou):
            r.update(model=model, variant=variant)
            tape_records.append(r)

    pd.DataFrame(glob_records).to_csv(args.out, index=False)
    pd.DataFrame(tape_records).to_csv(args.per_tape_out, index=False)
    print(f"✅ global metrics → {args.out}")
    print(f"✅ per-tape metrics → {args.per_tape_out}")


if __name__ == "__main__":
    main()
