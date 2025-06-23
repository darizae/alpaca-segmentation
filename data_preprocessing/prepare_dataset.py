#!/usr/bin/env python3
# ------------------------------------------------------------------
# prepare_dataset.py    – v1.1  (2025-06-05)
# ------------------------------------------------------------------
"""
Prepare an Alpaca-hum corpus for ANIMAL-SPOT.

Usage
-----
    python prepare_dataset.py  <corpus_root>

The corpus root must already contain `corpus_index.json`
(created by `build_index.py`).  All behaviour switches live in the
neighbouring JSON file `dataset_prep_configs.json`.
"""
from __future__ import annotations
import argparse, json, random, shutil, csv, re
from collections import defaultdict, Counter
from itertools import count
from pathlib import Path
from typing import List, Tuple, Dict
import soundfile as sf
import numpy as np


# ─────────────────────────── helpers ────────────────────────────
def ms(t: float) -> int:
    return int(round(t * 1000))


def build_name(cls: str, lbl: str, uid: int, year: str,
               tape: str, s_ms: int, e_ms: int) -> str:
    return f"{cls}-{lbl}_{uid:06d}_{year}_{tape}_{s_ms:07d}_{e_ms:07d}.wav"


class Interval:
    __slots__ = ("s", "e")

    def __init__(self, s: float, e: float):
        self.s, self.e = s, e

    def __lt__(self, other):  # for sort()
        return (self.s, self.e) < (other.s, other.e)


def find_free_slot(dur: float, container: Interval,
                   occupied: List['Interval'], margin: float,
                   rng: random.Random,
                   n_trials: int = 100) -> Tuple[float, float] | None:
    blocks = [Interval(max(container.s, iv.s - margin),
                       min(container.e, iv.e + margin))
              for iv in occupied]
    free: List[Tuple[float, float]] = []
    cur = container.s
    for iv in sorted(blocks):
        if iv.s - cur >= dur:
            free.append((cur, iv.s))
        cur = max(cur, iv.e)
    if container.e - cur >= dur:
        free.append((cur, container.e))
    if not free:
        return None
    a, b = rng.choice(free)
    start = rng.uniform(a, b - dur)
    return (start, start + dur)


# ────────────── split-strategy implementations ────────────────
def split_random_by_clip(items, seed, fracs):
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    i1 = int(fracs[0] * n);
    i2 = i1 + int(fracs[1] * n)
    return {"train": items[:i1], "val": items[i1:i2], "test": items[i2:]}


def split_proportional_by_tape(items, seed, fracs):
    rng = random.Random(seed)
    tape2 = defaultdict(list)
    for it in items:
        tape2[it["tape_key"]].append(it)
    tapes = list(tape2.keys());
    rng.shuffle(tapes)
    need = {k: int(f * len(items)) for k, f in zip(("train", "val", "test"), fracs)}
    splits = {k: [] for k in need}
    for tp in tapes:
        lack = {k: need[k] - len(v) for k, v in splits.items()}
        target = max(lack, key=lack.get)
        splits[target].extend(tape2[tp])
    return splits


def split_quality_balanced(items, seed, fracs):
    rng = random.Random(seed)
    q2 = defaultdict(list)
    for it in items:
        q2[it["quality"]].append(it)
    splits = {"train": [], "val": [], "test": []}
    for q, bucket in q2.items():
        bucket_split = split_random_by_clip(bucket, rng.randint(0, 1 << 30), fracs)
        for k in splits: splits[k].extend(bucket_split[k])
    for k in splits: rng.shuffle(splits[k])
    return splits


STRATEGY_FUN = {
    "random_by_clip": split_random_by_clip,
    "proportional_by_tape": split_proportional_by_tape,
    "quality_balanced": split_quality_balanced,
}

# ─────────────────────────── regexes ────────────────────────────
RAW_RE = re.compile(r"^(?P<animal>\w+?)_(?P<date>\d{8})(?:_[^_]+)?_cut\.wav$")
CLIP_RE = re.compile(r"^(?P<raw>.+?\.wav)_(?P<c0>\d+)_(?P<c1>\d+)\.wav$")


# ─────────────────────────── main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_root", type=Path)
    args = ap.parse_args()
    corpus = args.corpus_root.resolve()

    # sanity
    index_fp = corpus / "corpus_index.json"
    if not index_fp.exists():
        raise FileNotFoundError("Run build_index.py first – corpus_index.json missing.")

    cfg_fp = Path(__file__).with_name("dataset_prep_configs.json")
    with cfg_fp.open() as fh:
        cfg = json.load(fh)
    active_key = cfg["active_strategy"]
    s_cfg = cfg["strategies"][active_key]
    strat_name = s_cfg["split_strategy"]

    if strat_name not in STRATEGY_FUN:
        raise ValueError(f"Unknown split strategy {strat_name}")

    rng = random.Random(s_cfg["seed"])
    out_dir = corpus / f"dataset_{active_key}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    with index_fp.open() as fh:
        idx = json.load(fh)
    entries = idx["entries"]
    meta_clip_len = idx["meta"].get("clip_duration_s")

    # build occupied map only for tapes that really exist
    tape2hums = defaultdict(list)
    for e in entries:
        if e["quality"] < s_cfg["min_quality"] or e["raw_path"] is None:
            continue
        tape2hums[e["raw_path"]].append(Interval(e["raw_start_s"], e["raw_end_s"]))

    uid_gen = count(1)
    wav_rows = []
    exported = Counter()

    for e in entries:
        if e["quality"] < s_cfg["min_quality"]:
            continue

        # ---------- resolve raw metadata even if raw_path is None
        if e["raw_path"]:
            raw_fname = Path(e["raw_path"]).name
            raw_fp_exists = (corpus / e["raw_path"]).exists()
        else:
            clip_fname = Path(e["clip_path"]).name
            cm = CLIP_RE.match(clip_fname)
            raw_fname = cm["raw"]
            raw_fp_exists = False  # cannot fallback to raw

        year = RAW_RE.match(raw_fname)["date"][:4] if RAW_RE.match(raw_fname) else "0000"
        tape_clean = raw_fname.replace(".wav", "").replace("_", "-")

        # paths
        clip_fp = corpus / e["clip_path"]
        hum_fp = corpus / e["hum_path"]

        # ---------- export TARGET
        uid = next(uid_gen)
        tgt_fn = build_name("target", f"Q{e['quality']}", uid, year, tape_clean,
                            ms(e["clip_start_s"]), ms(e["clip_end_s"]))
        shutil.copy2(hum_fp, out_dir / tgt_fn)
        wav_rows.append({"fn": tgt_fn,
                         "tape_key": e["raw_uid"],
                         "quality": e["quality"]})
        exported["target"] += 1

        # ---------- export NOISE(S)
        for _ in range(s_cfg["noise_per_hum"]):
            dur = e["clip_end_s"] - e["clip_start_s"]
            margin = s_cfg["noise_mining"]["margin_s"]

            # try inside labelled 15-min clip
            clip_len = meta_clip_len or e["clip_dur_s"]
            slot = find_free_slot(dur,
                                  Interval(0, clip_len),
                                  [Interval(e["clip_start_s"], e["clip_end_s"])],
                                  margin, rng)
            source_fp = clip_fp if slot else None

            # optional fallback to raw
            if slot is None and s_cfg["noise_mining"]["fallback_raw"] and raw_fp_exists:
                raw_fp = corpus / e["raw_path"]
                info = sf.info(raw_fp)
                slot = find_free_slot(dur,
                                      Interval(0, info.frames / info.samplerate),
                                      tape2hums[e["raw_path"]],
                                      0.0, rng)
                source_fp = raw_fp if slot else None
            if slot is None:
                continue

            s_sec, e_sec = slot
            info = sf.info(source_fp)
            audio, _ = sf.read(source_fp,
                               start=int(s_sec * info.samplerate),
                               stop=int(e_sec * info.samplerate))
            uid_n = next(uid_gen)
            ns_fn = build_name("noise", "bg", uid_n, year, tape_clean,
                               ms(s_sec), ms(e_sec))
            sf.write(out_dir / ns_fn, audio, info.samplerate)
            wav_rows.append({"fn": ns_fn,
                             "tape_key": e["raw_uid"],
                             "quality": e["quality"]})
            exported["noise"] += 1

    # ---------- deterministic split
    fracs = (0.70, 0.15, 0.15)
    splits = STRATEGY_FUN[strat_name](wav_rows, s_cfg["seed"], fracs)
    for split, rows in splits.items():
        with (out_dir / f"{split}.csv").open("w", newline="") as fh:
            csv.writer(fh).writerows([[r["fn"]] for r in rows])

    # ---------- summary
    print(f"\n✅  strategy '{strat_name}' → {out_dir.relative_to(corpus)}")
    print(f"   {exported['target']} targets  +  {exported['noise']} noise clips")
    for s in ("train", "val", "test"):
        print(f"   {s:<5}: {len(splits[s]):5d} wavs")


# -----------------------------------------------------------------
if __name__ == "__main__":
    main()
