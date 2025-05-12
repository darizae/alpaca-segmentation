# alpaca_data_loader/checks.py
"""End-to-end sanity checks for the Alpaca hum corpus.

Run as script
-------------
```
python -m alpaca_data_loader.checks           # uses ./data
python -m alpaca_data_loader.checks --data /custom/path/to/data
```
If *any* assertion fails the script exits with code 1, making it CI-friendly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

from . import DATA_DIR, load_dataframe


# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _assert(condition: bool, msg: str):
    if not condition:
        raise AssertionError(msg)


def _all_exist(paths: Iterable[Path], what: str):
    missing = [p for p in paths if not p.exists()]
    _assert(not missing, f"{what}: {len(missing)} missing files (first: {missing[:3]})")


# ---------------------------------------------------------------------------
# Individual checks ----------------------------------------------------------
# ---------------------------------------------------------------------------

def check_png_files(df: pd.DataFrame, data_dir: Path):
    """Every PNG must exist, load, and be RGB/8-bit."""
    print("⏳  PNG existence & integrity …", end=" ")
    pngs = [data_dir / p for p in df.spec_png_path]
    _all_exist(pngs, "PNG")
    for p in pngs[:8]:  # spot-check a handful
        im = Image.open(p)
        _assert(im.mode == "RGB", f"{p} not RGB")
        for ch_min, ch_max in im.getextrema():
            _assert(0 <= ch_min <= 255 and 0 <= ch_max <= 255,
                    f"{p} not 8-bit range")
    print("✓")


def check_npy_files(df: pd.DataFrame, data_dir: Path):
    """Every .npy tensor exists and contains finite floats."""
    print("⏳  NPY existence & content …", end=" ")
    npys = [data_dir / p for p in df.spec_path]
    _all_exist(npys, "NPY")
    for p in np.random.choice(npys, size=min(5, len(npys)), replace=False):
        arr = np.load(p)
        _assert(np.isfinite(arr).all(), f"NaN/Inf in {p}")
    print("✓")


def check_time_consistency(df: pd.DataFrame):
    print("⏳  Timestamp geometry …", end=" ")
    # 1) raw‐timeline sanity
    _assert((df.hum_start_s < df.hum_end_s).all(),
            "hum_start ≥ hum_end detected")

    # 2) pick out the rows that *do* live (or at least start) inside their clip
    in_clip = df.hum_start_rel_clip_s.notna()
    clip_len = df.clip_end_s - df.clip_start_s

    # now both sides have identical indexes = where in_clip is True
    _assert((df.loc[in_clip, "hum_start_rel_clip_s"] >= 0).all(),
            "Negative start inside clip")
    _assert((df.loc[in_clip, "hum_end_rel_clip_s"]
             <= clip_len.loc[in_clip] + 1e-3).all(),
            "Hum extends beyond clip window")
    print("✓")


def check_quality_range(df: pd.DataFrame):
    print("⏳  Quality codes …", end=" ")
    _assert(df.quality.between(1, 5).all(), "quality ∉ [1,5]")
    print("✓")


def check_uid_uniqueness(df: pd.DataFrame):
    print("⏳  UID uniqueness …", end=" ")
    _assert(df.uid.is_unique, "Duplicate uid values")
    print("✓")


def check_foreign_keys(hums: pd.DataFrame, clips: pd.DataFrame, raws: pd.DataFrame):
    print("⏳  Foreign-key links …", end=" ")
    _assert(set(hums.clip_uid) <= set(clips.uid), "Unknown clip_uid in hums")
    _assert(set(hums.raw_uid) <= set(raws.uid), "Unknown raw_uid in hums")
    print("✓")


# ---------- NEW: dataset size & file-count coherence -----------------------

def check_dataset_size(hums: pd.DataFrame, clips: pd.DataFrame, raws: pd.DataFrame, *, data_dir: Path):
    """Ensure the various counts across folders / indices line up."""
    print("⏳  File counts vs indices …", end=" ")

    seg_wavs = list((data_dir / "segmented_wav_onlyhums").glob("*.wav"))
    lbl_wavs = list((data_dir / "labelled_recordings").glob("*.wav"))
    raw_wavs = list((data_dir / "raw_recordings").glob("*.wav"))
    pngs = list((data_dir / "spec_png_cache_rgb8").glob("*.png"))
    npys = list((data_dir / "spec_cache").glob("*.npy"))

    _assert(len(hums) == len(seg_wavs) == len(pngs) == len(npys),
            f"Hum rows / wav / png / npy mismatch: {len(hums)}, {len(seg_wavs)}, {len(pngs)}, {len(npys)}")

    # labelled clips: allow 0–1 difference (some clips may fail parsing)
    _assert(abs(len(clips) - len(lbl_wavs)) <= 1,
            f"Clip index vs files: {len(clips)} vs {len(lbl_wavs)}")

    # raw wavs: we expect actual audio files ≤ raw rows (placeholders inflate)
    _assert(len(raw_wavs) <= len(raws), "raw_recordings folder has more files than index_raw rows?")
    _assert(len(raw_wavs) >= 1, "No raw recordings present")

    print("✓")


# ---------------------------------------------------------------------------
# Master entry ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def run_all(data_dir: Path | None = None):
    ddir = Path(data_dir) if data_dir else DATA_DIR

    hums = load_dataframe("hums", data_dir=ddir)
    clips = load_dataframe("clips", data_dir=ddir)
    raws = load_dataframe("raw", data_dir=ddir)

    check_png_files(hums, ddir)
    check_npy_files(hums, ddir)
    check_time_consistency(hums)
    check_quality_range(hums)
    check_uid_uniqueness(hums)
    check_foreign_keys(hums, clips, raws)
    check_dataset_size(hums, clips, raws, data_dir=ddir)

    print("\n🎉  All sanity checks passed.")


# ---------------------------------------------------------------------------
# CLI wrapper ----------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run Alpaca dataset sanity checks")
    ap.add_argument("--data", type=Path, default=None,
                    help="Path to the data folder (defaults to repo/data)")
    args = ap.parse_args()

    try:
        run_all(args.data)
    except AssertionError as e:
        print(f"❌  Sanity check failed: {e}")
        sys.exit(1)
