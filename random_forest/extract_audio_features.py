#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import soundfile as sf

from audio_features.robust_features import raven_robust_features
from audio_features.mfcc_features import mfcc_summary

SELTAB = Path("cnn_candidates.csv")  # your merged selection table from the CNN
AUDIO_ROOT = Path("/path/to/wavs")  # base dir containing wave files

OUT_DIR = Path("feature_extraction_py");
OUT_DIR.mkdir(exist_ok=True)

# 1) Load selections
df = pd.read_csv(SELTAB)

# 2) For each wave, cache audio to avoid re-reading
cache = {}


def load_wave(fname):
    fname = AUDIO_ROOT / fname
    if fname not in cache:
        y, sr = sf.read(fname, always_2d=False)
        if y.ndim > 1:  # multi-channel → mixdown
            y = y.mean(axis=1)
        cache[fname] = (y.astype("float32"), sr)
    return cache[fname]


robust_rows, mfcc_rows, union_rows = [], [], []
for _, r in df.iterrows():
    wave = r["wave_file"]
    t0, t1 = float(r["Begin time (s)"]), float(r["End time (s)"])
    fmin = float(r.get("Low Freq (Hz)", 0.0))
    fmax = float(r.get("High Freq (Hz)", 0.0)) or None

    y, sr = load_wave(wave)

    spec = raven_robust_features(y, sr, t0, t1, fmin=fmin, fmax=fmax)
    mfc = mfcc_summary(y, sr, t0, t1)

    base = {"wave_file": wave, "selection": int(r["selection"])}
    robust_rows.append({**base, **spec})
    mfcc_rows.append({**base, **mfc})
    union_rows.append({**base, **spec, **mfc})

robust = pd.DataFrame(robust_rows)
mfcc = pd.DataFrame(mfcc_rows)
union = pd.DataFrame(union_rows)

robust.to_csv(OUT_DIR / "spectral_robust_py.csv", index=False)
mfcc.to_csv(OUT_DIR / "mfcc_py.csv", index=False)
union.to_csv(OUT_DIR / "features_py.csv", index=False)
print("✓ Wrote:", (OUT_DIR / "spectral_robust_py.csv").resolve())
