# tests/test_prepared_dataset.py
import re, soundfile as sf
from pathlib import Path

PAT = re.compile(
    r"^(?P<class>target|noise)-(?P<info>[^_]+)_\d{6}_(?P<year>\d{4})_(?P<tape>[^_]+)_(?P<s>\d{7})_(?P<e>\d{7})\.wav$")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "prepared"


def _clips():
    return list(DATA_DIR.glob("*.wav"))


def test_files_exist():
    assert _clips(), "prepared/ is empty – run make_alpaca_clips.py first"


def test_filename_pattern():
    bad = [p.name for p in _clips() if not PAT.match(p.name)]
    assert not bad, f"{len(bad)} filenames don’t follow ANIMAL-SPOT pattern"


def test_duration_positive():
    for p in _clips():
        info = sf.info(p)
        assert info.frames > 0, f"{p} has 0 frames"


def test_sr_consistency():
    srs = {sf.info(p).samplerate for p in _clips()}
    # allow either 48 k or 44.1 k, but not a mess of random SRs
    assert srs.issubset({48000, 44100}), f"Unexpected sample rates: {srs}"


def test_noise_ratio():
    n_targets = sum(1 for p in _clips() if p.name.startswith("target"))
    n_noise = sum(1 for p in _clips() if p.name.startswith("noise"))
    assert n_noise >= n_targets, "Need at least 1 noise per target"
