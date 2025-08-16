#!/usr/bin/env python3
"""
build_metrics_summary.py
Aggregate *all* rf_…_report.txt files in this directory and emit a
Markdown summary with tables, rankings and brief conclusions.

Usage
-----
    cd random_forest/metrics
    python build_metrics_summary.py
"""

from pathlib import Path
import re, textwrap, statistics as st
import csv

METRIC_PATHS = list(Path(".").glob("rf_*_report.txt"))
OUT_MD = Path("rf_run_summary.md")
OUT_CSV = Path("rf_run_summary.csv")

# ──────────────────────────────────────────────────────────────────
# 1.  regex helpers
# ──────────────────────────────────────────────────────────────────
RE_HEADER = re.compile(r"Random Forest on (?P<csv>\S+)\s+\(neg_per_pos=(?P<neg>[\w]+)\)")
RE_ROCAUC = re.compile(r"ROC-AUC\s*:\s*(?P<roc>[0-9.]+)")
RE_PRAUC = re.compile(r"PR-AUC\s*:\s*(?P<pr>[0-9.]+)")
RE_CM = re.compile(
    r"\[\[\s*(\d+)\s+(\d+)\s*\]\s*[\r\n\s]*\[\s*(\d+)\s+(\d+)\s*\]\]", re.S
)
RE_CLASS1 = re.compile(r"^\s*1\s+(?P<p>[0-9.]+)\s+(?P<r>[0-9.]+)\s+(?P<f>[0-9.]+)", re.M)


def parse_metrics_text(txt: str) -> dict:
    """Return dict of the key numbers we care about."""
    out = {}
    m = RE_HEADER.search(txt)
    if m:
        out["csv"] = m.group("csv")
        out["neg"] = m.group("neg")
    out["roc_auc"] = float(RE_ROCAUC.search(txt).group("roc"))
    out["pr_auc"] = float(RE_PRAUC.search(txt).group("pr"))
    cm = RE_CM.search(txt)
    if cm:
        tn, fp, fn, tp = map(int, cm.groups())
    else:
        # fallback: pick first four ints after the header
        hdr = txt.find("Confusion matrix")
        ints = re.findall(r"\d+", txt[hdr:])[:4] if hdr != -1 else []
        if len(ints) == 4:
            tn, fp, fn, tp = map(int, ints)
        else:
            raise ValueError("Could not parse confusion matrix in metrics file.")

    out.update(dict(tn=tn, fp=fp, fn=fn, tp=tp))
    c1 = RE_CLASS1.search(txt)
    out["prec1"] = float(c1.group("p"))
    out["rec1"] = float(c1.group("r"))
    out["f1_1"] = float(c1.group("f"))
    # derived metrics
    out["accuracy"] = (out["tp"] + out["tn"]) / sum(out[k] for k in ("tp", "tn", "fp", "fn"))
    return out


def load_all():
    for path in METRIC_PATHS:
        with path.open() as fh:
            yield path.name, parse_metrics_text(fh.read())


runs = list(load_all())
if not runs:
    raise SystemExit("No rf_*_report.txt files found here.")

# ──────────────────────────────────────────────────────────────────
# 2.  ranking (by F1 of target class, tie-break by PR-AUC)
# ──────────────────────────────────────────────────────────────────
runs.sort(key=lambda r: (r[1]["f1_1"], r[1]["pr_auc"]), reverse=True)

# ──────────────────────────────────────────────────────────────────
# 3.  build Markdown
# ──────────────────────────────────────────────────────────────────
md = ["# Random-Forest experiment summary\n"]
md.append("Parsed *{}* runs located in `metrics/`.\n".format(len(runs)))


def fmt_float(x): return f"{x:0.3f}"


# main comparison table
md.append("## Comparison table\n")
md.append("| rank | csv | neg/pos | F1(target) | Precision | Recall | PR-AUC | ROC-AUC | FP | FN | accuracy |")
md.append("|-----:|:----|:-------:|-----------:|-----------:|-------:|-------:|--------:|--:|--:|---------:|")

for rank, (fname, r) in enumerate(runs, 1):
    md.append(
        f"| {rank} | {r['csv']} | {r['neg']} | {fmt_float(r['f1_1'])} | "
        f"{fmt_float(r['prec1'])} | {fmt_float(r['rec1'])} | "
        f"{fmt_float(r['pr_auc'])} | {fmt_float(r['roc_auc'])} | "
        f"{r['fp']} | {r['fn']} | {fmt_float(r['accuracy'])} |"
    )

# global stats
f1s = [r["f1_1"] for _, r in runs]
prauc = [r["pr_auc"] for _, r in runs]
md.append("\n## Overall observations\n")
md.append(textwrap.dedent(f"""
* **Best F1** : {max(f1s):0.3f};  **median F1** : {st.median(f1s):0.3f}
* **Best PR-AUC** : {max(prauc):0.3f};  **median PR-AUC** : {st.median(prauc):0.3f}

A higher noise-down-sampling ratio (`neg_per_pos=1`) unsurprisingly lifts recall
and F1 but costs precision.  The MFCC+Spectral feature set (`features_with_labels.csv`)
tends to dominate PR-AUC, suggesting the hand-engineered stats do add signal
beyond MFCCs alone.

If field time is scarce, the **top-ranked run (rank 1)** offers the best balance:
precision ≈ {runs[0][1]['prec1']:0.2f} with recall ≈ {runs[0][1]['rec1']:0.2f}.
For broader monitoring (accepting more false alarms) consider the run ranked 2.
"""))

OUT_MD.write_text("\n".join(md))
print(f"✓ Markdown summary written → {OUT_MD.resolve()}")

# ──────────────────────────────────────────────────────────────────
# 3b.  build CSV
# ──────────────────────────────────────────────────────────────────
csv_headers = [
    "rank", "csv", "neg/pos", "F1(target)", "Precision", "Recall",
    "ROC-AUC", "FP", "FN", "accuracy"
]

with OUT_CSV.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(csv_headers)
    for rank, (fname, r) in enumerate(runs, 1):
        writer.writerow([
            rank,
            r['csv'],
            r['neg'],
            round(r['f1_1'], 3),
            round(r['prec1'], 3),
            round(r['rec1'], 3),
            round(r['roc_auc'], 3),
            r['fp'],
            r['fn'],
            round(r['accuracy'], 3),
        ])

print(f"✓ CSV summary written → {OUT_CSV.resolve()}")

