from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MASK_DIR = ROOT / "data/MaSTr1325/masks"

paths = sorted(MASK_DIR.glob("*.png"))
print("looking in:", MASK_DIR)
print("exists:", MASK_DIR.exists(), "| files found:", len(paths))

mask = np.array(Image.open(paths[0]))
print("shape:", mask.shape, "dtype:", mask.dtype)
print("values in this mask:", np.unique(mask))

counts = {}
for letter in ["a", "b", "a", "a", "b", "c"]:
    counts[letter] = counts.get(letter, 0) + 1
print(counts)

totals = {}
for p in paths:
    arr = np.array(Image.open(p))
    values, counts = np.unique(arr, return_counts=True)
    for v, c in zip(values, counts):
        totals[int(v)] = totals.get(int(v), 0) + int(c)

all_pixels = sum(totals.values())
for v, c in sorted(totals.items()):
    print(f"value {v}: {100 * c / all_pixels:.2f}%")




from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


class MaSTr(Dataset):
    def __init__(self, split_file):
        self.pairs = []
        for line in open(ROOT / split_file):
            img_path, mask_path = line.strip().split(",")
            self.pairs.append((ROOT / img_path, ROOT / mask_path))

        self.lut = np.arange(256, dtype=np.uint8)
        self.lut[4] = 255

    def __len__(self):
        length = len (self.pairs)
        return length

    def __getitem__(self, i):
        img_path, mask_path = self.pairs[i]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        mask = self.lut[mask]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255
        mask = torch.from_numpy(mask).long()
        return img, mask


if __name__ == "__main__":
    ds = MaSTr("splits/train.txt")
    print("samples:", len(ds))                  
    img, mask = ds[0]
    print("img:", img.shape, img.dtype)         
    print("mask:", mask.shape, mask.dtype)      
    print("mask values:", torch.unique(mask))   

