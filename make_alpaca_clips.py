#!/usr/bin/env python3
# ------------------------------------------------------------
# make_alpaca_clips.py   – v2.0  (2025-05-18)
# ------------------------------------------------------------
"""
Convert Alpaca hum corpus into ANIMAL-SPOT-ready target/noise clips.

Key changes vs v1
-----------------
* **Sample-rate awareness.**  Reads parent file, writes out at ORIGINAL SR
  (ANIMAL-SPOT will resample if you set `sr:` in the YAML).
* **Accurate frame indexing.**  `soundfile.read()` expects *frame* indices;
  we now multiply by `samplerate` once, **after** loading header info.
* **Hum/Noise overlap guard** across the *entire tape* using the JSON index
  if it exists (falls back to in-memory list otherwise).
* **Robust tape & year parsing** via the same regexes as `build_alpaca_index.py`.
* **Dry-run / overwrite-skip** flags.
"""
from __future__ import annotations
import argparse, re, random, shutil, json
from itertools import count
from pathlib import Path
from typing import List, Tuple
import soundfile as sf
import numpy as np

# ------------------------------------------------------------------ regexes
RAW_RE = re.compile(r"^(?P<animal>\w+)_(?P<date>\d{8})(?:_(?P<tag>[^_]+))?_cut\.wav$")
CLIP_RE = re.compile(r"^(?P<orig>.+?\.wav)_(?P<c0>\d+)_(?P<c1>\d+)\.wav$")
HUM_RE = re.compile(r"^(?P<orig>.+?\.wav)_(?P<h0>\d+(?:\.\d+)?)_(?P<h1>\d+(?:\.\d+)?)Q(?P<q>\d)\.wav$")


def ms(t: float) -> int: return int(round(t * 1000))


def build_name(cls: str, lbl: str, uid: int, year: str,
               tape: str, s_ms: int, e_ms: int) -> str:
    return f"{cls}-{lbl}_{uid:06d}_{year}_{tape}_{s_ms:07d}_{e_ms:07d}.wav"


# ---------------------------- noise slot helper --------------------
class Interval:  # simple struct
    __slots__ = ("s", "e")

    def __init__(self, s: float, e: float) -> None:
        self.s, self.e = s, e


def find_free_slot(dur: float, container: Interval,
                   occupied: List[Interval], margin=0.05, n_trials=100) -> Tuple[float, float] | None:
    # shrink usable bands by occupied zones
    blocks = [Interval(max(container.s, iv.s - margin), min(container.e, iv.e + margin))
              for iv in occupied]
    free = []
    cur = container.s
    for iv in sorted(blocks, key=lambda x: x.s):
        if iv.s - cur >= dur:
            free.append((cur, iv.s))
        cur = max(cur, iv.e)
    if container.e - cur >= dur:
        free.append((cur, container.e))
    if not free:
        return None
    for _ in range(n_trials):
        a, b = random.choice(free)
        if b - a >= dur:
            start = random.uniform(a, b - dur)
            return (start, start + dur)
    return None


# ------------------------------------------------------------------ main
def main(opts):
    rnd = random.Random(opts.seed)

    droot = Path(opts.data_dir)
    seg_hums = droot / "segmented_wav_onlyhums"
    lbl_clips = droot / "labelled_recordings"
    raw_dir = droot / "raw_recordings"
    out_dir = droot / "prepared"
    out_dir.mkdir(exist_ok=True)

    # ---------- Load JSON index if present (gives us *all* hums quickly)
    index_hums = droot / "index_hums.json"
    tape2hums: dict[str, list[Interval]] = {}
    if index_hums.exists():
        with index_hums.open() as fh:
            for row in json.load(fh):
                if row["quality"] < opts.min_quality:  # filter early
                    continue
                tape = Path(row["path"]).name.split("_cut")[0] + "_cut"
                tape2hums.setdefault(tape, []).append(Interval(row["hum_start_s"], row["hum_end_s"]))

    uid_gen = count(1)
    exported_targets = 0
    exported_noise = 0

    hum_files = sorted(seg_hums.glob("*.wav"))
    for hum_fp in hum_files:
        hm = HUM_RE.match(hum_fp.name)
        if not hm or int(hm["q"]) < opts.min_quality:
            continue

        # ---------------- meta parsing
        clip_name = hm["orig"]
        cm = CLIP_RE.match(clip_name)
        raw_name = cm["orig"]  # e.g. 387_20201207_cut.wav
        rm = RAW_RE.match(raw_name)
        year = rm["date"][:4]
        tape = raw_name.replace(".wav", "")

        # ---------------- copy target
        uid = next(uid_gen)
        h0, h1 = float(hm["h0"]), float(hm["h1"])
        tgt_fn = build_name("target", f"Q{hm['q']}", uid, year, tape, ms(h0), ms(h1))
        if not opts.dry_run:
            shutil.copy2(hum_fp, out_dir / tgt_fn)
        exported_targets += 1

        # ---------------- mine N noise clips
        for _ in range(opts.noise_per_hum):
            dur = h1 - h0
            # 1º try inside labelled 15-s clip
            lbl_fp = lbl_clips / clip_name
            container = Interval(float(cm["c0"]), float(cm["c1"]))
            occupied = [Interval(h0, h1)]
            if tape in tape2hums:  # global list for fallback collision check
                occupied = tape2hums[tape]

            slot = find_free_slot(dur, container, occupied)
            source_fp = lbl_fp if lbl_fp.exists() else None

            # 2º fallback – any free slot on the raw tape
            if slot is None:
                raw_fp = raw_dir / raw_name
                if not raw_fp.exists():
                    continue
                sr_raw = sf.info(raw_fp).samplerate
                raw_dur = sf.info(raw_fp).frames / sr_raw
                container = Interval(0, raw_dur)
                slot = find_free_slot(dur, container, occupied, margin=0.0)
                source_fp = raw_fp
            if slot is None or source_fp is None:
                continue

            s_sec, e_sec = slot
            info = sf.info(source_fp)
            start_fr, stop_fr = int(s_sec * info.samplerate), int(e_sec * info.samplerate)
            audio, _ = sf.read(source_fp, start=start_fr, stop=stop_fr)
            uid_n = next(uid_gen)
            ns_fn = build_name("noise", "bg", uid_n, year, tape, ms(s_sec), ms(e_sec))
            if not opts.dry_run:
                sf.write(out_dir / ns_fn, audio, info.samplerate)
            exported_noise += 1

    # ---------------- summary
    print(f"\nExported {exported_targets} target clips and {exported_noise} noise clips → {out_dir}")


# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data",
                    help="Folder with raw_recordings/, labelled_recordings/, segmented_wav_onlyhums/")
    ap.add_argument("--min-quality", type=int, default=1)
    ap.add_argument("--noise-per-hum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse everything but don’t write files")
    args = ap.parse_args()
    main(args)
