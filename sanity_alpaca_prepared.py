#!/usr/bin/env python3
# ------------------------------------------------------------
# sanity_alpaca_prepared.py
# ------------------------------------------------------------
"""
Print corpus statistics for data/prepared/   (target/noise clips).
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import soundfile as sf
import numpy as np
from collections import Counter, defaultdict

PAT = re.compile(
    r"^(?P<class>target|noise)-(?P<info>[^_]+)_\d{6}_(?P<year>\d{4})_(?P<tape>[^_]+)_(?P<s>\d{7})_(?P<e>\d{7})\.wav$")


def main(d):
    paths = list(Path(d).glob("*.wav"))
    if not paths:
        print("✖  No wav files found.")
        sys.exit(1)

    cls_counts = Counter()
    qual_counts = Counter()
    lens = []
    sr_set = set()
    tapes = defaultdict(int)

    for p in paths:
        m = PAT.match(p.name)
        if not m:
            print(f"!  bad filename: {p.name}")
            continue
        cls = m["class"]
        cls_counts[cls] += 1
        if cls == "target":
            qual_counts[m["info"]] += 1
        tapes[m["tape"]] += 1
        s, e = int(m["s"]), int(m["e"])
        lens.append((e - s) / 1000)
        sr_set.add(sf.info(p).samplerate)

    print("── Corpus summary ──")
    print(f" files:      {len(paths):,}")
    print(f" targets:    {cls_counts['target']:,}")
    print(f" noise:      {cls_counts['noise']:,}  "
          f"(ratio {cls_counts['noise'] / max(1, cls_counts['target']):.2f} per target)")
    print(f" quality:    " + ", ".join(f"{q}:{n}" for q, n in qual_counts.items()))
    print(f" duration s: min {np.min(lens):.2f} | mean {np.mean(lens):.2f} | max {np.max(lens):.2f}")
    print(f" sampleRate: {sorted(sr_set)}")
    print(f" tapes:      {len(tapes)} unique\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/prepared")
    args = ap.parse_args()
    main(args.data_dir)
