from __future__ import annotations
import numpy as np
import librosa


# or: from spafe.features.mfcc import mfcc as spafe_mfcc   # if you prefer spafe

def mfcc_summary(
        y, sr, t0, t1, n_mfcc=13, n_fft=2048, hop_length=512, include_deltas=True
) -> dict:
    """
    Frame-level MFCCs inside [t0, t1], then summarise (mean, std) per coefficient.
    Returns a flat dict with keys: mfcc1_mean, mfcc1_std, ..., (and deltas if requested)
    """
    i0 = max(0, int(round(t0 * sr)));
    i1 = min(len(y), int(round(t1 * sr)))
    seg = y[i0:i1]
    if seg.size == 0:
        return {}

    M = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    feats = {}

    def add_stats(mat, prefix):
        feats.update({f"{prefix}{i + 1}_mean": float(mat[i].mean()) for i in range(mat.shape[0])})
        feats.update({f"{prefix}{i + 1}_std": float(mat[i].std(ddof=1) if mat.shape[1] > 1 else 0.0)
                      for i in range(mat.shape[0])})

    add_stats(M, "mfcc")
    if include_deltas:
        D1 = librosa.feature.delta(M, order=1)
        D2 = librosa.feature.delta(M, order=2)
        add_stats(D1, "d_mfcc")
        add_stats(D2, "dd_mfcc")
    return feats
