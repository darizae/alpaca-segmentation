#!/usr/bin/env python3
"""
Extract robust spectral stats + MFCC summaries for all selections
in a folder of Raven/Animal-Spot selection tables, and write a single
combined CSV: data/spectral_feature_annotations/py/features_py.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import re
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm

from audio_features import raven_robust_features, mfcc_summary


# ------------------------------ CLI args ------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sel-dir", type=Path, default=Path("data/cnn_predictions/annotated"),
                   help="Folder containing selection-table .txt files")
    p.add_argument("--audio-dir", type=Path, default=Path("data/labelled_recordings"),
                   help="Folder containing the corresponding WAV files")
    p.add_argument("--out-dir", type=Path, default=Path("data/spectral_feature_annotations/py"),
                   help="Destination folder for the output CSVs")
    # STFT / MFCC params
    p.add_argument("--n-fft", type=int, default=2048, help="STFT FFT size")
    p.add_argument("--hop", type=int, default=1024, help="STFT hop length")
    p.add_argument("--n-mfcc", type=int, default=13, help="Number of MFCC coefficients")
    p.add_argument("--no-deltas", action="store_true", help="Disable Δ and ΔΔ MFCCs")
    return p.parse_args()


# ------------------------------ selection-table parsing ------------------------------

COL_ALIASES = {
    "Selection": "selection",
    "selection": "selection",
    "Begin time (s)": "Begin time (s)",
    "Begin Time (s)": "Begin time (s)",
    "End time (s)": "End time (s)",
    "End Time (s)": "End time (s)",
    "Low Freq (Hz)": "Low Freq (Hz)",
    "Low Frequency (Hz)": "Low Freq (Hz)",
    "High Freq (Hz)": "High Freq (Hz)",
    "High Frequency (Hz)": "High Freq (Hz)",
    "Sound type": "Sound type",
    "Sound Type": "Sound type",
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {c: COL_ALIASES[c] for c in df.columns if c in COL_ALIASES}
    out = df.rename(columns=ren)
    required = {"selection", "Begin time (s)", "End time (s)"}
    if not required.issubset(out.columns):
        missing = required - set(out.columns)
        raise ValueError(f"Selection table missing required columns: {missing}")
    return out


def wave_from_table_name(path: Path) -> str:
    """
    Map selection-table filename -> WAV filename.
    e.g. 'ABC.wav_0_3550_selection.txt' -> 'ABC.wav_0_3550.wav'
    """
    stem = Path(path.name).stem
    stem = re.sub(r'_selection$', '', stem, flags=re.IGNORECASE)  # drop trailing '_selection'
    return f"{stem}.wav"


# ------------------------------ audio loading ------------------------------

class AudioCache:
    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[Path, tuple[np.ndarray, int]] = {}

    def load(self, wave_name: str) -> tuple[np.ndarray, int]:
        wav_path = self.root / wave_name
        if wav_path in self._cache:
            return self._cache[wav_path]
        if not wav_path.exists():
            alt = wav_path.with_suffix(".WAV")
            if alt.exists():
                wav_path = alt
            else:
                raise FileNotFoundError(f"WAV not found for selection table: {wav_path}")
        y, sr = sf.read(wav_path, always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32, copy=False)
        self._cache[wav_path] = (y, sr)
        return y, sr


# ------------------------------ main ------------------------------

def main():
    args = parse_args()
    sel_dir: Path = args.sel_dir
    audio_dir: Path = args.audio_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(p for p in sel_dir.glob("*.txt") if p.is_file())
    if not txt_files:
        print(f"No selection tables found in: {sel_dir}", file=sys.stderr)
        sys.exit(1)

    cache = AudioCache(audio_dir)

    union_rows = []
    n_err = 0

    print(f"Scanning {len(txt_files)} selection tables in {sel_dir} …")
    for txt in tqdm(txt_files, unit="file"):
        try:
            df = pd.read_csv(txt, sep="\t")
        except Exception as e:
            n_err += 1
            print(f"[WARN] Failed to read {txt.name}: {e}", file=sys.stderr)
            continue

        try:
            df = normalise_columns(df)
        except Exception as e:
            n_err += 1
            print(f"[WARN] Skipping {txt.name}: {e}", file=sys.stderr)
            continue

        wave_file = wave_from_table_name(txt)
        try:
            y, sr = cache.load(wave_file)
        except Exception as e:
            n_err += 1
            print(f"[WARN] Missing audio for {txt.name}: {e}", file=sys.stderr)
            continue

        has_label = "Sound type" in df.columns
        label_series = (df["Sound type"].astype(str).str.strip().str.lower()
                        if has_label else None)

        for i, row in df.iterrows():
            try:
                sel = int(row["selection"])
                t0 = float(row["Begin time (s)"])
                t1 = float(row["End time (s)"])
                fmin = float(row["Low Freq (Hz)"]) if "Low Freq (Hz)" in row and row["Low Freq (Hz)"] != "" else None
                fmax = float(row["High Freq (Hz)"]) if "High Freq (Hz)" in row and row["High Freq (Hz)"] != "" else None
                if fmax is not None and fmax <= 0:
                    fmax = None
            except Exception as e:
                n_err += 1
                print(f"[WARN] Bad row in {txt.name} (index {i}): {e}", file=sys.stderr)
                continue

            spec = raven_robust_features(
                y, sr, t0, t1, fmin=fmin, fmax=fmax,
                n_fft=args.n_fft, hop_length=args.hop, window="hann", center=True
            )
            mfc = mfcc_summary(
                y, sr, t0, t1,
                n_mfcc=args.n_mfcc, n_fft=args.n_fft, hop_length=args.hop,
                include_deltas=not args.no_deltas
            )

            base = {"wave_file": wave_file, "selection": sel}
            if has_label:
                lbl = label_series.iloc[i]
                if lbl in {"target", "noise"}:
                    base["validated_label"] = lbl

            union_rows.append({**base, **spec, **mfc})

    # Build and write the single combined CSV
    df_all = pd.DataFrame(union_rows)
    key_cols = ["wave_file", "selection"]
    lbl_col = ["validated_label"] if "validated_label" in df_all.columns else []
    df_all = df_all[key_cols + lbl_col + [c for c in df_all.columns if c not in key_cols + lbl_col]]

    out_all = out_dir / "features_py.csv"
    df_all.to_csv(out_all, index=False)

    print(f"\n✓ Wrote: {out_all.resolve()}")
    if n_err:
        print(f"Completed with {n_err} warning(s). See stderr for details.")


if __name__ == "__main__":
    main()
