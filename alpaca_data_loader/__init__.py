"""Light‑weight utility package for working with the Alpaca hum dataset.

The package assumes the following **project layout** (already present):

```
project_root/
├── data/
│   ├── index_raw.json
│   ├── index_clips.json
│   ├── index_hums.json
│   ├── spec_cache/               # *.npy  (log‑mel tensors)
│   └── spec_png_cache_rgb8/      # *.png  (RGB, 8‑bit)
└── alpaca_data_loader/
    └── __init__.py               # ← you are here
```

Import patterns
---------------
```python
from alpaca_data_loader import load_dataframe, AlpacaDataset

hums = load_dataframe("hums")
train_ds = AlpacaDataset(hums, transform=my_transforms)
```

The loader keeps **zero business logic** in Jupyter notebooks and is easy to
re‑use in scripts or a future CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Union

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T

__all__ = [
    "DATA_DIR",
    "load_dataframe",
    "AlpacaDataset",
]

# ---------------------------------------------------------------------------
# 1.  Default data directory (project_root/data)
# ---------------------------------------------------------------------------

#: Path to the folder that contains all indices & caches.
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# 2.  Helper – load one of the JSON index files into a DataFrame
# ---------------------------------------------------------------------------

def _index_name_to_file(which: Literal["raw", "clips", "hums"]) -> Path:
    return DATA_DIR / f"index_{which}.json"


def load_dataframe(which: Literal["raw", "clips", "hums"],
                   columns: Optional[list[str]] = None,
                   data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Return a *pandas* DataFrame for one of the index files.

    Parameters
    ----------
    which : {"raw", "clips", "hums"}
        Name of the table to load.
    columns : list[str] | None
        Optional list of columns to keep (saves memory if you only need a few).
    data_dir : Path | None
        Override the default :pydata:`DATA_DIR` – useful when running outside
        the repository root.
    """
    dir_ = data_dir or DATA_DIR
    path = dir_ / f"index_{which}.json"
    if not path.exists():
        raise FileNotFoundError(f"Index not found: {path}")
    with path.open() as fh:
        data = json.load(fh)
    df = pd.DataFrame(data)
    return df[columns] if columns else df


# ---------------------------------------------------------------------------
# 3.  Torch Dataset – returns (image_tensor, target_dict)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Entry:
    """Row wrapper for **index_hums.json** with *both* timelines."""

    uid: int
    spec_png_path: str
    spec_path: str
    path: str  # wav path relative to data dir

    # clip‑relative (maybe None if hum outside 15‑s window)
    hum_start_rel_clip_s: Optional[float]
    hum_end_rel_clip_s: Optional[float]

    # raw‑timeline (always present)
    hum_start_s: float
    hum_end_s: float

    # misc
    quality: int
    animal_id: str
    date: str
    clip_uid: int
    raw_uid: int
    clip_start_s: float
    clip_end_s: float

    @classmethod
    def from_row(cls, r: pd.Series) -> "_Entry":
        return cls(
            uid=r.uid,
            spec_png_path=r.spec_png_path,
            spec_path=r.spec_path,
            path=r.path,
            hum_start_rel_clip_s=r.hum_start_rel_clip_s,
            hum_end_rel_clip_s=r.hum_end_rel_clip_s,
            hum_start_s=r.hum_start_s,
            hum_end_s=r.hum_end_s,
            quality=r.quality,
            animal_id=r.animal_id,
            date=r.date,
            clip_uid=r.clip_uid,
            raw_uid=r.raw_uid,
            clip_start_s=r.clip_start_s,
            clip_end_s=r.clip_end_s,
        )


class AlpacaDataset(Dataset):
    """PyTorch dataset that yields **(spectrogram, targets)**.

    The spectrogram is loaded from pre‑rendered **RGB, 8‑bit PNGs** and then
    passed through the *transform* callable (defaults to ImageNet‑style
    resizing and normalisation).

    *Targets* are returned as a dict so you can plug in custom losses:
    ``{"t_start": tensor, "t_end": tensor, "quality": tensor}``.
    """

    def __init__(
            self,
            df: Union[pd.DataFrame, str | Path],
            data_dir: Optional[Path] = None,
            transform: Optional[torch.nn.Module] = None,
    ) -> None:
        if isinstance(df, (str, Path)):
            df = load_dataframe("hums", data_dir=data_dir)
        # keep only needed columns for speed
        self._entries = [_Entry.from_row(r) for _, r in df.iterrows()]
        self.data_dir = data_dir or DATA_DIR
        self.transform = transform or self._imagenet_default_transform()

    # ------------------------------------------------------------------
    @staticmethod
    def _imagenet_default_transform() -> torch.nn.Module:
        # Gray viridis PNGs are already RGB – we just resize & normalise.
        return T.Compose([
            T.Resize((224, 224), antialias=True),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int):
        entry = self._entries[idx]

        # -------- image --------
        img = Image.open(self.data_dir / entry.spec_png_path).convert("RGB")
        x = self.transform(img)

        # raw‐timeline targets (always present)
        t_start_raw = torch.tensor(entry.hum_start_s, dtype=torch.float32)
        t_end_raw = torch.tensor(entry.hum_end_s, dtype=torch.float32)

        # clip‐relative targets, might be None → use NaN to indicate “outside clip”
        if entry.hum_start_rel_clip_s is not None:
            t_start_clip = torch.tensor(entry.hum_start_rel_clip_s, dtype=torch.float32)
            t_end_clip = torch.tensor(entry.hum_end_rel_clip_s, dtype=torch.float32)
        else:
            t_start_clip = torch.tensor(float("nan"))
            t_end_clip = torch.tensor(float("nan"))

        # -------- target (what the loss needs) --------
        target = {
            "t_start_raw": t_start_raw,
            "t_end_raw": t_end_raw,
            "t_start_clip": t_start_clip,
            "t_end_clip": t_end_clip,
            "quality": torch.tensor(entry.quality, dtype=torch.int64),
            "loss_weight": torch.tensor(1.0 + 0.2 * (5 - entry.quality),
                                        dtype=torch.float32)  # example weighting
        }

        # -------- meta (for bookkeeping / evaluation) --------
        meta = {
            "uid": entry.uid,
            "animal_id": entry.animal_id,
            "date": entry.date,
            "clip_uid": entry.clip_uid,
            "raw_uid": entry.raw_uid,
            "clip_start_s": entry.clip_start_s,
            "clip_end_s": entry.clip_end_s,
            "hum_start_s": entry.hum_start_s,
            "hum_end_s": entry.hum_end_s,
            "spec_path": str(self.data_dir / entry.spec_path),
            "wav_path": str(self.data_dir / entry.path),
        }

        return x, target, meta


# ---------------------------------------------------------------------------
# 4.  Convenience function – quick train/val split by `animal_id`
# ---------------------------------------------------------------------------

def train_val_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Return *index* arrays for train/val, stratified by animal_id."""
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["animal_id"]))
    return train_idx, val_idx
