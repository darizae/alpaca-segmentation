"""
Create 8-bit, RGB-viridis spectrogram PNGs and update index_hums.json
"""

from pathlib import Path
import json, numpy as np, imageio.v2 as iio, matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps
from tqdm.auto import tqdm

# ----------------------------- paths ------------------------------
project_root = Path(__file__).resolve().parent.parent
BASE = project_root / "data"
PNG_DIR = BASE / "spec_png_cache_rgb8"
PNG_DIR.mkdir(exist_ok=True)

DB_MIN, DB_MAX = -80.0, 0.0  # global dB window

# ----- 65 536-colour viridis lookup table --------------------------
viridis = colormaps.get_cmap("viridis").resampled(65_536)   # resample gives desired LUT size
viridis_lut = (viridis(np.linspace(0, 1, 65_536))[:, :3] * 65535).astype(np.uint16)


# ------------------ iterate over hum entries ----------------------
hums = json.loads((BASE / "index_hums.json").read_text())

for h in tqdm(hums, desc="PNG16"):
    uid = h["uid"]
    npy_path = BASE / h["spec_path"]
    png_rel = f"{PNG_DIR.name}/{uid}.png"
    png_path = BASE / png_rel
    h["spec_png_path"] = png_rel

    if png_path.exists():
        continue

    spec = np.load(npy_path)  # (mels, frames), dB
    img16 = np.clip((spec - DB_MIN) * 65535 / (DB_MAX - DB_MIN), 0, 65535
                    ).astype(np.uint16)  # greyscale 16-bit
    rgb = viridis_lut[img16]  # (H, W, 3), uint16

    rgb8 = (rgb // 257).astype(np.uint8)  # integer down-shift
    iio.imwrite(str(png_path), rgb8, format="PNG-PIL")  # no warning

# save back the updated metadata
(BASE / "index_hums.json").write_text(json.dumps(hums, indent=2))
print(f"Saved {len(hums)} PNGs in {PNG_DIR}")
