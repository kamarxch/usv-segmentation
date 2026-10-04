import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

from dataset import MaSTr, CLASS_NAMES
from metrics import (NUM_CLASSES, update_confusion, iou_from_confusion,
                     precision_recall_from_confusion)
from model import build_model

ROOT = Path(__file__).resolve().parents[1]
device = "cuda" if torch.cuda.is_available() else "cpu"


BUCKETS = [("small (<500 px)", 0, 500),
           ("medium (500-5000 px)", 500, 5000),
           ("large (>=5000 px)", 5000, float("inf"))]


def upsample(logits, size):
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def run(model, loader):
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    buckets = {name: {"components": 0, "gt_pixels": 0, "found_pixels": 0} for name, _, _ in BUCKETS}

    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            logits = upsample(model(pixel_values=img).logits, mask.shape[-2:])
            pred = logits.argmax(dim=1)
            update_confusion(conf, pred, mask)

            for p, m in zip(pred.cpu().numpy(), mask.cpu().numpy()):
                labeled, n = ndimage.label(m == 0)           
                if n == 0:
                    continue
                flat = labeled.ravel()
                areas = np.bincount(flat)[1:]                 
                found = np.bincount(flat, weights=(p == 0).ravel().astype(np.float64))[1:]
                for area, f in zip(areas, found):
                    for name, lo, hi in BUCKETS:
                        if lo <= area < hi:
                            buckets[name]["components"] += 1
                            buckets[name]["gt_pixels"] += int(area)
                            buckets[name]["found_pixels"] += int(f)
    return conf, buckets


def plot_confusion(conf, path):
    conf = conf.cpu().numpy().astype(float)
    norm = conf / conf.sum(1, keepdims=True).clip(min=1)      
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{norm[i, j]:.3f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    ds = MaSTr(f"splits/{args.split}.txt", train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model()
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device, weights_only=True))
    model.to(device).eval()

    conf, buckets = run(model, loader)
    iou = iou_from_confusion(conf)
    precision, recall = precision_recall_from_confusion(conf)

    print(f"\n=== {args.split} set, {len(ds)} images, checkpoint {args.ckpt} ===")
    print(f"mIoU: {iou.mean().item():.4f}")
    for k, name in enumerate(CLASS_NAMES):
        print(f"{name:9s} IoU {iou[k]:.4f}  precision {precision[k]:.4f}  recall {recall[k]:.4f}")

    print("\nRecall of class-0 pixels by size of the connected region:")
    bucket_out = {}
    for name, s in buckets.items():
        rec = s["found_pixels"] / s["gt_pixels"] if s["gt_pixels"] else None
        bucket_out[name] = {**s, "recall": rec}
        shown = f"{rec:.4f}" if rec is not None else "n/a"
        print(f"  {name:22s} regions {s['components']:5d}  recall {shown}")

    results = {
        "split": args.split,
        "checkpoint": args.ckpt,
        "num_images": len(ds),
        "miou": iou.mean().item(),
        "per_class": {n: {"iou": iou[k].item(), "precision": precision[k].item(),
                          "recall": recall[k].item()} for k, n in enumerate(CLASS_NAMES)},
        "class0_recall_by_region_size": bucket_out,
    }
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results" / f"{args.split}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_confusion(conf, ROOT / "results" / "figures" / f"confusion_matrix_{args.split}.png")
    print(f"\nsaved results/{args.split}_metrics.json and results/figures/confusion_matrix_{args.split}.png")


if __name__ == "__main__":
    main()