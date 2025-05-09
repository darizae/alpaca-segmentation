#!/usr/bin/env python
# build_alpaca_index.py  – v1.1  (handles missing raw recordings)
# ---------------------------------------------------------------
"""
Index the Alpaca audio corpus into three JSON tables
(raw, labelled 15-s clips, cropped hums) with rich metadata.
For hum segments the script also caches a log-mel spectrogram (.npy).

If a labelled clip references a raw recording that you don’t possess,
a *placeholder* Raw entry with type="raw_missing" is generated so that
UID relationships stay intact and downstream code never breaks.
"""

from __future__ import annotations
import json, re, math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Union
import itertools
import numpy as np
import soundfile as sf
import librosa
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# CONFIG – edit once, shared by the entire repo
# ---------------------------------------------------------------------

SPEC_CFG = dict(
    n_fft=2002,  # ← 41.7 ms @ 48 kHz
    hop_length=1001,  # ← 50 % overlap
    n_mels=64,  # keep or raise to 128 if you want finer detail
    fmin=0,
    fmax=4000,  # hums live below 4 kHz
    power=2.0,
)
SAMPLE_RATE_TARGET = 48_000  # all hums will be resampled to this


# ---------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------

@dataclass
class Base:
    uid: int
    type: str
    path: str
    animal_id: str
    date: str
    extra_tag: Optional[str]
    duration_s: float
    samplerate: int
    channels: int
    rms_db: Optional[float] = None
    max_db: Optional[float] = None
    snr_db: Optional[float] = None
    dataset_split: Optional[str] = None
    spec_path: Optional[str] = None
    spec_png_path: Optional[str] = None
    mask_path: Optional[str] = None
    # frozen spectrogram params
    n_fft: int = SPEC_CFG["n_fft"]
    hop_length: int = SPEC_CFG["hop_length"]
    n_mels: int = SPEC_CFG["n_mels"]
    fmin: int = SPEC_CFG["fmin"]
    fmax: int = SPEC_CFG["fmax"]


@dataclass
class Raw(Base):
    pass


@dataclass(kw_only=True)
class Clip(Base):
    raw_uid: int
    clip_start_s: float
    clip_end_s: float


@dataclass(kw_only=True)
class Hum(Base):
    raw_uid: int
    clip_uid: int
    clip_start_s: float
    clip_end_s: float
    hum_start_s: float
    hum_end_s: float
    hum_start_rel_clip_s: float
    hum_end_rel_clip_s: float
    quality: int


# ---------------------------------------------------------------------
# regex patterns
# ---------------------------------------------------------------------

RAW_PAT = re.compile(
    r"""^(?P<animal>\w+)_(?P<date>\d{8})(?:_(?P<tag>[^_]+))?_cut\.wav$""", re.VERBOSE
)
LABEL_PAT = re.compile(
    r"""^(?P<orig>.+?\.wav)_(?P<clip_start>\d+)_(?P<clip_end>\d+)\.wav$""", re.VERBOSE
)
HUM_PAT = re.compile(
    r"""^(?P<orig>.+?\.wav)_(?P<hum_start>\d+(?:\.\d+)?)_(?P<hum_end>\d+(?:\.\d+)?)Q(?P<q>\d)\.wav$""",
    re.VERBOSE,
)

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
DIRS = {
    "raw": data_dir / "raw_recordings",
    "clips": data_dir / "labelled_recordings",
    "hums": data_dir / "segmented_wav_onlyhums",
}
SPEC_CACHE = data_dir / "spec_cache"
SPEC_CACHE.mkdir(exist_ok=True)

_uid = itertools.count(1)


def next_uid() -> int: return next(_uid)


UID_RAW: Dict[str, int] = {}  # raw filename → uid
UID_CLIP: Dict[str, int] = {}  # clip filename → uid

RAWS, CLIPS, HUMS = [], [], []  # will be filled on the fly


def wav_header(path: Path):
    info = sf.info(path)
    return info.frames / info.samplerate, info.samplerate, info.channels


def db_stats(y: np.ndarray):
    rms = np.sqrt(np.mean(y ** 2))
    peak = np.max(np.abs(y))
    rms_db = 20 * math.log10(rms + 1e-12)
    peak_db = 20 * math.log10(peak + 1e-12)
    return rms_db, peak_db, peak_db - rms_db


def save_spec(y: np.ndarray, sr: int, uid: int) -> str:
    if sr != SAMPLE_RATE_TARGET:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE_TARGET)
        sr = SAMPLE_RATE_TARGET
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, power=SPEC_CFG["power"],
        **{k: SPEC_CFG[k] for k in ("n_fft", "hop_length", "n_mels", "fmin", "fmax")}
    )
    logS = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    out = SPEC_CACHE / f"{uid}.npy"
    np.save(out, logS)
    return str(out.relative_to(data_dir))


# -----  missing-raw helper  -------------------------------------------------

def ensure_raw_uid(raw_name: str) -> int:
    """Return UID of raw file; create placeholder if raw not on disk."""
    if raw_name in UID_RAW:
        return UID_RAW[raw_name]

    m = RAW_PAT.match(raw_name)
    if m is None:
        raise ValueError(f"Cannot infer metadata from raw filename: {raw_name}")
    g = m.groupdict()
    uid = next_uid()
    placeholder = Raw(
        uid=uid,
        type="raw_missing",
        path=f"MISSING/{raw_name}",
        animal_id=g["animal"],
        date=f"{g['date'][:4]}-{g['date'][4:6]}-{g['date'][6:]}",
        extra_tag=g["tag"],
        duration_s=float("nan"),
        samplerate=-1,
        channels=-1,
    )
    RAWS.append(placeholder)
    UID_RAW[raw_name] = uid
    return uid


# ---------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------

def parse_raw(path: Path):
    m = RAW_PAT.match(path.name)
    if not m:
        raise ValueError(path)
    g = m.groupdict()
    dur, sr, ch = wav_header(path)
    uid = next_uid()
    RAWS.append(
        Raw(
            uid=uid,
            type="raw",
            path=str(path.relative_to(data_dir)),
            animal_id=g["animal"],
            date=f"{g['date'][:4]}-{g['date'][4:6]}-{g['date'][6:]}",
            extra_tag=g["tag"],
            duration_s=dur,
            samplerate=sr,
            channels=ch,
        )
    )
    UID_RAW[path.name] = uid


def parse_clip(path: Path):
    m = LABEL_PAT.match(path.name)
    if not m:
        raise ValueError(path)
    g = m.groupdict()
    raw_name = g["orig"]
    raw_uid = ensure_raw_uid(raw_name)  # <-- placeholder if needed
    raw_g = RAW_PAT.match(raw_name).groupdict()

    dur, sr, ch = wav_header(path)
    uid = next_uid()
    CLIPS.append(
        Clip(
            uid=uid,
            type="labelled_clip",
            path=str(path.relative_to(data_dir)),
            animal_id=raw_g["animal"],
            date=f"{raw_g['date'][:4]}-{raw_g['date'][4:6]}-{raw_g['date'][6:]}",
            extra_tag=raw_g["tag"],
            duration_s=dur,
            samplerate=sr,
            channels=ch,
            raw_uid=raw_uid,
            clip_start_s=float(g["clip_start"]),
            clip_end_s=float(g["clip_end"]),
        )
    )
    UID_CLIP[path.name] = uid


def parse_hum(path: Path):
    m = HUM_PAT.match(path.name)
    if not m:
        raise ValueError(path)
    g = m.groupdict()
    clip_name = g["orig"]
    if clip_name not in UID_CLIP:
        # clip not yet seen (can happen if clips folder missing) – create stub
        parse_clip(DIRS["clips"] / clip_name)

    clip_uid = UID_CLIP[clip_name]
    clip_g = LABEL_PAT.match(clip_name).groupdict()
    raw_name = clip_g["orig"]
    raw_uid = ensure_raw_uid(raw_name)
    raw_g = RAW_PAT.match(raw_name).groupdict()

    y, sr = sf.read(path)
    rms_db, max_db, snr_db = db_stats(y)
    uid = next_uid()
    spec_rel = save_spec(y, sr, uid)

    HUMS.append(
        Hum(
            uid=uid,
            type="hum_segment",
            path=str(path.relative_to(data_dir)),
            animal_id=raw_g["animal"],
            date=f"{raw_g['date'][:4]}-{raw_g['date'][4:6]}-{raw_g['date'][6:]}",
            extra_tag=raw_g["tag"],
            duration_s=len(y) / sr,
            samplerate=sr,
            channels=1 if y.ndim == 1 else y.shape[1],
            rms_db=rms_db,
            max_db=max_db,
            snr_db=snr_db,
            spec_path=spec_rel,
            raw_uid=raw_uid,
            clip_uid=clip_uid,
            clip_start_s=float(clip_g["clip_start"]),
            clip_end_s=float(clip_g["clip_end"]),
            hum_start_s=float(g["hum_start"]),
            hum_end_s=float(g["hum_end"]),
            hum_start_rel_clip_s=float(g["hum_start"]) - float(clip_g["clip_start"]),
            hum_end_rel_clip_s=float(g["hum_end"]) - float(clip_g["clip_start"]),
            quality=int(g["q"]),
        )
    )


# ---------------------------------------------------------------------
# build index
# ---------------------------------------------------------------------

def build():
    for p in tqdm(sorted(DIRS["raw"].glob("*.wav")), desc="raw"):
        parse_raw(p)
    for p in tqdm(sorted(DIRS["clips"].glob("*.wav")), desc="clips"):
        parse_clip(p)
    for p in tqdm(sorted(DIRS["hums"].glob("*.wav")), desc="hums"):
        parse_hum(p)

    out_map = [
        ("index_raw.json", RAWS),
        ("index_clips.json", CLIPS),
        ("index_hums.json", HUMS),
    ]
    for fname, lst in out_map:
        with open(data_dir / fname, "w", encoding="utf-8") as fh:
            json.dump([asdict(e) for e in lst], fh, indent=2)
        print(f"Wrote {fname}  ({len(lst)} entries)")


if __name__ == "__main__":
    build()
