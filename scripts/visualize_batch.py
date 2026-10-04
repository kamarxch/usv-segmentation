import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))      
from dataset import MaSTr, IMAGENET_MEAN, IMAGENET_STD

ds = MaSTr("splits/train.txt", train=True)

mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for k in range(4):
    img, mask = ds[k * 100]              

    img_to_show = (img * std + mean).clamp(0, 1).permute(1, 2, 0)

    axes[0, k].imshow(img_to_show)
    axes[1, k].imshow(mask, cmap="tab10", vmin=0, vmax=9)
    axes[0, k].axis("off")
    axes[1, k].axis("off")

plt.tight_layout()
plt.savefig(ROOT / "results/figures/augmented_samples.png", dpi=100)
print("saved results/figures/augmented_samples.png")