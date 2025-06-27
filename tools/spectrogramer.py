#!/usr/bin/env python3
# spectrogramer.py  –  v1.0  (2025-06-27)

"""
Generate PNG spectrograms from one of three index-file flavours

  · corpus_index.json       (before dataset prep)
  · variant_index.json      (after dataset prep)
  · index.json              (prediction log)

Usage
-----
    python spectrogramer.py \
        --index /path/to/index.json \
        --audio-root /path/to/wavs_root \
        --out /tmp/spectrograms \
        --config spectrogram_configs.json
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import soundfile as sf
from scipy.signal import stft
import matplotlib as mpl
from PIL import Image


# ──────────────────── basic fast STFT → PNG ─────────────────────
def fast_spectrogram_png(
        y: np.ndarray,
        sr: int,
        png_path: Path,
        cfg: Dict[str, Any],
) -> None:
    nperseg = min(cfg["n_fft"], len(y))
    hop = min(cfg["hop_length"], max(nperseg - 1, 1))
    f, t, Z = stft(
        y,
        fs=sr,
        nperseg=nperseg,
        noverlap=nperseg - hop,
        window=cfg["window"],
        padded=False,
        boundary=None,
    )

    S_db = 20 * np.log10(np.abs(Z) + 1e-6)
    vmax = S_db.max()
    vmin = vmax - cfg.get("dyn_range_db", 80)
    S_db = np.clip(S_db, vmin, vmax)
    norm = (S_db - vmin) / (vmax - vmin)

    cmap = mpl.colormaps["magma"]
    rgba = cmap(norm, bytes=True)
    rgb = rgba[..., :3]

    # crop to desired frequency band
    f_lo = max(cfg.get("freq_low", 100), 1)
    f_hi = cfg.get("freq_high", sr // 2)
    i0 = np.searchsorted(f, f_lo, side="left")
    i1 = np.searchsorted(f, f_hi, side="right")
    rgb = rgb[i0:i1, :, :]

    # optional: resize for consistent width
    pps = cfg.get("pixels_per_second", 300)
    if pps > 0:
        from PIL import Image
        w_target = int(len(t) * hop / sr * pps)
        img = Image.fromarray(rgb).resize(
            (w_target, rgb.shape[0]), resample=Image.BILINEAR
        )
    else:
        img = Image.fromarray(rgb)

    img.save(png_path, format="PNG", optimize=True)


# ──────────────────── helpers to locate audio ───────────────────
def load_index(fp: Path) -> Dict[str, Any]:
    with fp.open() as f:
        return json.load(f)


def which_index_flavour(idx: Dict[str, Any]) -> str:
    sample = idx["entries"][0]
    if "hum_path" in sample:  # corpus_index.json
        return "corpus"
    if "class" in sample:  # variant_index.json
        return "variant"
    if "pred_path" in sample:  # prediction index.json
        return "pred"
    raise ValueError("Unrecognised index schema")


def iter_audio_segments(
        idx: Dict[str, Any],
        flavour: str,
        audio_root: Path,
) -> Iterable[tuple[str, np.ndarray, int]]:
    """
    Yields (png_stub, mono_audio, sample_rate)

    png_stub is used to name the PNG ('.png' will be appended).
    """
    if flavour == "corpus":  # one PNG per hum clip
        for e in idx["entries"]:
            wav_fp = audio_root / e["hum_path"]
            y, sr = sf.read(wav_fp)
            yield wav_fp.stem, y, sr

    elif flavour == "variant":  # e["fn"] lives beside dataset
        for e in idx["entries"]:
            wav_fp = audio_root / e["fn"]
            y, sr = sf.read(wav_fp)
            yield wav_fp.stem, y, sr

    elif flavour == "pred":  # slice from full tape
        for e in idx["entries"]:
            wav_fp = audio_root / e["tape"]
            info = sf.info(wav_fp)
            y, _ = sf.read(
                wav_fp,
                start=int(e["start_s"] * info.samplerate),
                stop=int(e["end_s"] * info.samplerate),
            )
            yield f"{wav_fp.stem}_{e['uid']}", y, info.samplerate


# ──────────────────────────── main ──────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--audio-root", required=True, type=Path,
                    help="Folder that actually holds the WAV files")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None,
                    help="spectrogram_configs.json (optional)")
    args = ap.parse_args()

    # 1) spectrogram config --------------------------------------------------
    if args.config and args.config.exists():
        cfg = json.loads(args.config.read_text())["presets"]["default"]
    else:  # sensible hard-coded default
        cfg = dict(
            n_fft=2048,
            hop_length=512,
            window="hann",
            dyn_range_db=80,
            freq_low=100,
            freq_high=4000,
            pixels_per_second=300,
        )

    # 2) output dir
    args.out.mkdir(parents=True, exist_ok=True)

    # 3) load index and detect flavour
    idx = load_index(args.index)
    flavour = which_index_flavour(idx)
    print(f"→ index flavour detected: {flavour}")

    # 4) iterate, render, save
    n_done = 0
    for stub, y, sr in iter_audio_segments(idx, flavour, args.audio_root):
        png_path = args.out / f"{stub}.png"
        fast_spectrogram_png(y, sr, png_path, cfg)
        n_done += 1

    print(f"✓ wrote {n_done} PNGs → {args.out}")


if __name__ == "__main__":
    main()
