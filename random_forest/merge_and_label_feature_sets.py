#!/usr/bin/env python3
"""
Merge MFCC & spectral features with manual labels from RavenLite selection
tables.  Produces three training sets:

    mfcc_with_labels.csv
    spectral_with_labels.csv
    features_with_labels.csv

Raises descriptive exceptions for:
  A. Unlabelled rectangles
  B. File-name mismatch
  C. Selection-number drift
  D. Extra predictions

Author: Daniel’s favourite AI side-kick   ☺
"""

from pathlib import Path
import pandas as pd
import re
import glob

# ------------------------------------------------------------------ #
#                         CONFIG SECTION
# ------------------------------------------------------------------ #
ANNOT_DIR = Path("/Users/danie/repos/alpaca-segmentation/random_forest/data/cnn_predictions/annotated")
MFCC_CSV = Path(
    "/Users/danie/repos/alpaca-segmentation/random_forest/data/spectral_feature_annotations/raw/MFCC_table.csv")
SPEC_CSV = Path(
    "/Users/danie/repos/alpaca-segmentation/random_forest/data/spectral_feature_annotations/raw/spectral_robust.csv")
OUT_DIR = MFCC_CSV.parent / "processed"  # will be created if absent
OUT_DIR.mkdir(exist_ok=True)

LABEL_FIELD = "Sound type"  # column that holds ’target’ | ’noise’
VALID_LABELS = {"target", "noise"}  # enforce exact spelling


# ------------------------------------------------------------------ #
#                   0.  helper functions & regexes
# ------------------------------------------------------------------ #
class MergeError(RuntimeError): pass


class UnlabelledRectangles(MergeError): pass  # A


class FileNameMismatch(MergeError): pass  # B


class SelectionNumberDrift(MergeError): pass  # C


class ExtraPredictions(MergeError): pass  # D


# --- regexes to extract ids from the MFCC 'Class' column -----------
_PATTERNS = [
    re.compile(r'^(?P<wave>.+?)_sel\.?(?P<sel>\d+)(?:[._].*)?$'),  # ..._sel.12...
    re.compile(r'^(?P<wave>.+)_(?P<sel>\d+)(?:[._].*)?$'),  # ..._12...
]


def _parse_class_field(s: str):
    core = s.strip().removesuffix('.wav').removesuffix('.WAV')
    for pat in _PATTERNS:
        m = pat.match(core)
        if m:
            return f"{m.group('wave')}.wav", int(m.group('sel'))
    raise MergeError(f"Cannot parse MFCC 'Class' value: {s!r}")


def _norm_wave(name: str) -> str:
    """
    Strip any path components and force the extension to lower-case '.wav'.

    Examples
    --------
    >>> _norm_wave("/some/dir/388_20250204_193000.WAV")
    '388_20250204_193000.wav'
    >>> _norm_wave("x_20250205_193000 2nd obs")
    'x_20250205_193000 2nd obs.wav'
    """
    leaf = Path(name.strip()).name        # keep only the file name
    base = Path(leaf).stem                # drop whatever extension exists
    return f"{base}.wav"


# ------------------------------------------------------------------ #
# 1.  read & tidy MFCC + spectral tables
# ------------------------------------------------------------------ #
print("Loading feature CSVs…")
mfcc_raw = pd.read_csv(MFCC_CSV)
spec_raw = pd.read_csv(SPEC_CSV)

# --- parse MFCC ids
mfcc_ids = pd.DataFrame(mfcc_raw['Class'].apply(_parse_class_field).tolist(),
                        columns=['wave_file', 'selection'])
mfcc_ids['wave_file'] = mfcc_ids['wave_file'].map(_norm_wave)
mfcc_ids['selection'] = mfcc_ids['selection'].astype(int)
mfcc = pd.concat([mfcc_ids, mfcc_raw.drop(columns=['Class'])], axis=1)

# --- tidy spectral ids
spec = spec_raw.rename(columns={'Begin File': 'wave_file', 'Selection': 'selection'})
spec['wave_file'] = spec['wave_file'].map(_norm_wave)
spec['selection'] = spec['selection'].astype(int)

# --- merge feature sets
features = mfcc.merge(spec, on=['wave_file', 'selection'], how='inner', validate='one_to_one')
if len(features) != len(mfcc):
    raise MergeError(f"Feature merge dropped rows: MFCC={len(mfcc)}, merged={len(features)}")

# ------------------------------------------------------------------ #
# 2.  ingest manual labels from every annotated txt
# ------------------------------------------------------------------ #
print("Collecting manual labels…")
lab_frames = []
for txt in glob.glob(str(ANNOT_DIR / "*.txt")):
    table = pd.read_csv(txt, sep="\t")
    if LABEL_FIELD not in table.columns:
        raise MergeError(f"File {txt} lacks column {LABEL_FIELD!r}")

    good = table[table[LABEL_FIELD].isin(VALID_LABELS)]
    if good.empty:
        raise UnlabelledRectangles(f"File {txt} contains no rows with valid labels {VALID_LABELS}")

    labels = good[["Selection", LABEL_FIELD]].copy()
    labels.rename(columns={"Selection": "selection", LABEL_FIELD: "validated_label"}, inplace=True)
    labels["selection"] = labels["selection"].astype(int)
    labels["wave_file"] = _norm_wave(Path(txt).stem + ".wav")
    lab_frames.append(labels)

all_labels = pd.concat(lab_frames, ignore_index=True)
print(f" → {len(all_labels)} labelled rectangles across {len(lab_frames)} tables")

# ------------------------------------------------------------------ #
# 3.  sanity checks B, C, D
# ------------------------------------------------------------------ #
# B. wave-file mismatch
missing_waves = set(all_labels['wave_file']) - set(features['wave_file'])
if missing_waves:
    raise FileNameMismatch(
        f"These txt files do not correspond to any WAV in the feature tables:\n"
        + "\n".join(sorted(missing_waves)))

# C. selection number drift — look per wav
drift = []
for w in sorted(all_labels['wave_file'].unique()):
    lab_sel = set(all_labels.loc[all_labels.wave_file == w, 'selection'])
    feat_sel = set(features.loc[features.wave_file == w, 'selection'])
    extra_lab = lab_sel - feat_sel
    extra_feat = feat_sel - lab_sel
    if extra_lab or extra_feat:
        drift.append((w, extra_lab, extra_feat))
if drift:
    msgs = []
    for w, el, ef in drift:
        msgs.append(f"{w}:  labels-only {sorted(el)[:5]}…  features-only {sorted(ef)[:5]}…")
    raise SelectionNumberDrift(
        "Selection numbers diverge between labels and features:\n" + "\n".join(msgs))

# D. extra predictions (unlabelled feature rows)
merged = features.merge(all_labels, on=['wave_file', 'selection'], how='left')
orphans = merged[merged['validated_label'].isna()]
if not orphans.empty:
    msg = (f"{len(orphans)} feature rows have *no* manual label "
           f"(examples: {orphans[['wave_file', 'selection']].head().to_dict('records')})")
    raise ExtraPredictions(msg)

# ------------------------------------------------------------------ #
# 4.  build the three output tables
# ------------------------------------------------------------------ #
mfcc_with_lab = mfcc.merge(all_labels, on=['wave_file', 'selection'], how='inner')
spec_with_lab = spec.merge(all_labels, on=['wave_file', 'selection'], how='inner')
features_with_lab = features.merge(all_labels, on=['wave_file', 'selection'], how='inner')

mfcc_with_lab.to_csv(OUT_DIR / "mfcc_with_labels.csv", index=False)
spec_with_lab.to_csv(OUT_DIR / "spectral_with_labels.csv", index=False)
features_with_lab.to_csv(OUT_DIR / "features_with_labels.csv", index=False)

print("✓ All three labelled feature files written to", OUT_DIR.resolve())
print("   mfcc_with_labels      :", mfcc_with_lab.shape)
print("   spectral_with_labels  :", spec_with_lab.shape)
print("   features_with_labels  :", features_with_lab.shape)
