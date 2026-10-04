from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = ["obstacle", "water", "sky"]
IGNORE_INDEX = 255

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MaSTr(Dataset):
    def __init__(self, split_file, train=False):
        self.pairs = []
        with open(ROOT / split_file) as f:
            for line in f:
                img_path, mask_path = line.strip().split(",")
                self.pairs.append((ROOT / img_path, ROOT / mask_path))

        self.lut = np.arange(256, dtype=np.uint8)
        self.lut[4] = IGNORE_INDEX

        norm = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        if train:
            self.tf = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                norm,
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([norm, ToTensorV2()])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, mask_path = self.pairs[i]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask = self.lut[mask]

        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].long()


if __name__ == "__main__":
    for name, train in [("train", True), ("val", False), ("test", False)]:
        ds = MaSTr(f"splits/{name}.txt", train=train)
        print(f"{name}: {len(ds)} samples")

    ds = MaSTr("splits/train.txt", train=True)
    img, mask = ds[0]
    print("img:", img.shape, img.dtype)
    print("img range:", round(img.min().item(), 2), round(img.max().item(), 2))
    print("mask:", mask.shape, mask.dtype)
    print("mask values:", torch.unique(mask))

    seen = set()
    for i in range(len(ds)):
        seen.update(torch.unique(ds[i][1]).tolist())
    print("all mask values in train:", sorted(seen))
    assert seen <= {0, 1, 2, IGNORE_INDEX}, "unexpected label value!"