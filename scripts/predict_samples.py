import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from dataset import MaSTr, IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD
from model import build_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# class colors for the plots: obstacle = red, water = blue, sky = light blue, ignore = black
PALETTE = {0: (220, 70, 60), 1: (40, 110, 200), 2: (150, 210, 240), IGNORE_INDEX: (0, 0, 0)}
MEAN = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
STD = np.array(IMAGENET_STD).reshape(3, 1, 1)


def colorize(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value, color in PALETTE.items():
        out[mask == value] = color
    return out


def to_rgb(img):
    """undo the normalization: (3, H, W) tensor -> (H, W, 3) array in [0, 1]"""
    return np.clip(img.numpy() * STD + MEAN, 0, 1).transpose(1, 2, 0)


@torch.no_grad()
def predict(model, img):
    logits = model(pixel_values=img.unsqueeze(0).to(device)).logits
    logits = F.interpolate(logits, size=img.shape[-2:], mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--n", type=int, default=6, help="how many of the worst images to plot")
    args = parser.parse_args()

    ds = MaSTr(f"splits/{args.split}.txt", train=False)
    model = build_model()
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device, weights_only=True))
    model.to(device).eval()

    # pass 1: count the wrongly predicted pixels of every image
    scores = []
    for i in range(len(ds)):
        img, mask = ds[i]
        m = mask.numpy()
        valid = m != IGNORE_INDEX
        errors = int(((predict(model, img) != m) & valid).sum())
        scores.append((errors, i))
    scores.sort(reverse=True)
    worst = scores[:args.n]
    print(f"{args.split}: mean errors per image {np.mean([s for s, _ in scores]):.0f} px, "
          f"median {np.median([s for s, _ in scores]):.0f} px")
    print("worst images:")
    for errors, i in worst:
        print(f"  {ds.pairs[i][0].name}  {errors} wrong pixels")

    # pass 2: plot photo / true mask / prediction / error map for the worst images
    fig, axes = plt.subplots(len(worst), 4, figsize=(16, 3.3 * len(worst)))
    axes = np.atleast_2d(axes)
    for ax, title in zip(axes[0], ["image", "ground truth", "prediction", "errors (red = wrong)"]):
        ax.set_title(title)
    for row, (errors, i) in zip(axes, worst):
        img, mask = ds[i]
        m = mask.numpy()
        pred = predict(model, img)
        err = (pred != m) & (m != IGNORE_INDEX)
        row[0].imshow(to_rgb(img))
        row[1].imshow(colorize(m))
        row[2].imshow(colorize(pred))
        row[3].imshow(err, cmap="Reds", vmin=0, vmax=1)
        row[0].set_ylabel(f"{ds.pairs[i][0].stem}\n{errors} px wrong", fontsize=8)
        for ax in row:
            ax.set_xticks([]); ax.set_yticks([])

    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    out_path = ROOT / "results" / "figures" / f"predictions_{args.split}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    print("saved", out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()