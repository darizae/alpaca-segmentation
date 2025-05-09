import numpy as np
import torchvision.transforms as T, torch
from PIL import Image
import json

with open("../data/index_hums.json") as fh:
    hums = json.load(fh)


def pil16_to_float(img):
    """PIL image (RGB, uint16) → torch.float32 in [0,1]"""
    arr = torch.from_numpy(np.array(img, dtype=np.uint16))
    return arr.float().permute(2, 0, 1) / 65535.0  # C×H×W


transform = T.Compose([
    T.Lambda(pil16_to_float),
    T.Resize((224, 224), antialias=True),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

img = transform(Image.open("../data/" + hums[0]["spec_png_path"]))
