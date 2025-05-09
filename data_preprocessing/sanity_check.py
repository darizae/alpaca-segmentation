from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
spec = np.load(data_dir / "spec_cache" / "286.npy")
print(spec.shape)  # ← (64, N_frames)
