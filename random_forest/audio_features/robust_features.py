from __future__ import annotations
import numpy as np
import librosa


def _slice_signal(y, sr, t0, t1):
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(y), int(round(t1 * sr)))
    return y[i0:i1]


def _stft_psd(y, sr, n_fft=2048, hop_length=512, win="hann", center=True):
    # STFT magnitude^2 = power spectral density proxy (unnormalized)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=win, center=center)
    PSD = (np.abs(S) ** 2).astype(np.float64)  # [freq, time]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(PSD.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)
    return PSD, freqs, times


def _band_mask(freqs, fmin, fmax):
    lo = 0 if fmin is None else np.searchsorted(freqs, max(0, fmin), side="left")
    hi = len(freqs) if fmax is None else np.searchsorted(freqs, fmax, side="right")
    return slice(lo, hi)


def _quantile_indexed(x, coords, q):
    """Quantile on a non-uniform axis: cumulative sum then search; linear interp."""
    x = np.clip(x, 0.0, np.inf)
    s = x.sum()
    if s <= 0:
        return float(coords[len(coords) // 2])
    pdf = x / s
    cdf = np.cumsum(pdf)
    idx = np.searchsorted(cdf, q, side="left")
    if idx == 0:
        return float(coords[0])
    if idx >= len(coords):
        return float(coords[-1])
    # linear interpolation between coords[idx-1] and coords[idx]
    c0, c1 = coords[idx - 1], coords[idx]
    w1 = (q - cdf[idx - 1]) / (cdf[idx] - cdf[idx - 1] + 1e-12)
    return float((1 - w1) * c0 + w1 * c1)


def _entropy_bits(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())


def raven_robust_features(
        y, sr, t0, t1, fmin=None, fmax=None, n_fft=2048, hop_length=512
) -> dict:
    """
    Compute Raven-style robust stats + entropies for a selection.

    Returns keys:
      'Dur 90% (s)', 'Dur 50% (s)',
      'Center Freq (Hz)', 'Freq 5% (Hz)', 'Freq 25% (Hz)', 'Freq 75% (Hz)', 'Freq 95% (Hz)',
      'BW 50% (Hz)', 'BW 90% (Hz)',
      'Avg Entropy (bits)', 'Agg Entropy (bits)'
    """
    seg = _slice_signal(y, sr, t0, t1)
    if seg.size == 0:
        return {k: np.nan for k in [
            "Dur 90% (s)", "Dur 50% (s)", "Center Freq (Hz)", "Freq 5% (Hz)",
            "Freq 25% (Hz)", "Freq 75% (Hz)", "Freq 95% (Hz)", "BW 50% (Hz)",
            "BW 90% (Hz)", "Avg Entropy (bits)", "Agg Entropy (bits)"]}

    PSD, freqs, times = _stft_psd(seg, sr, n_fft=n_fft, hop_length=hop_length)
    fsl = _band_mask(freqs, fmin, fmax)
    PSDb = PSD[fsl, :]  # band-limited PSD

    # Time envelope and time quantiles (t5, t25, t50, t75, t95)
    Et = PSDb.sum(axis=0)  # [time]
    t5 = _quantile_indexed(Et, times, 0.05)
    t25 = _quantile_indexed(Et, times, 0.25)
    t50 = _quantile_indexed(Et, times, 0.50)
    t75 = _quantile_indexed(Et, times, 0.75)
    t95 = _quantile_indexed(Et, times, 0.95)
    dur90 = max(0.0, t95 - t5)
    dur50 = max(0.0, t75 - t25)

    # Frequency spectrum (time-averaged PSD) & freq quantiles
    Sf = PSDb.mean(axis=1)  # [freq]
    f5 = _quantile_indexed(Sf, freqs[fsl], 0.05)
    f25 = _quantile_indexed(Sf, freqs[fsl], 0.25)
    f50 = _quantile_indexed(Sf, freqs[fsl], 0.50)  # center freq
    f75 = _quantile_indexed(Sf, freqs[fsl], 0.75)
    f95 = _quantile_indexed(Sf, freqs[fsl], 0.95)
    bw50 = max(0.0, f75 - f25)
    bw90 = max(0.0, f95 - f5)

    # Entropies
    # Avg entropy: mean over time of slice entropies within band
    Et_pos = PSDb.sum(axis=0) + 1e-12
    Pft = PSDb / Et_pos  # column-wise normalization
    Ht = (-(Pft * np.log2(np.clip(Pft, 1e-12, 1))).sum(axis=0))  # [time]
    avg_entropy = float(np.mean(Ht))

    # Agg entropy: entropy of the time-averaged spectrum
    Pf = Sf / (Sf.sum() + 1e-12)
    agg_entropy = _entropy_bits(Pf)

    return {
        "Dur 90% (s)": dur90,
        "Dur 50% (s)": dur50,
        "Center Freq (Hz)": f50,
        "Freq 5% (Hz)": f5,
        "Freq 25% (Hz)": f25,
        "Freq 75% (Hz)": f75,
        "Freq 95% (Hz)": f95,
        "BW 50% (Hz)": bw50,
        "BW 90% (Hz)": bw90,
        "Avg Entropy (bits)": avg_entropy,
        "Agg Entropy (bits)": agg_entropy,
    }
