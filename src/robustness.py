import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from dataset import MaSTr, IMAGENET_MEAN, IMAGENET_STD
from metrics import (NUM_CLASSES, update_confusion, iou_from_confusion,
                     precision_recall_from_confusion)
from model import build_model

ROOT = Path(__file__).resolve().parents[1]
device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)


def low_light(x, s):                       
    return x * s


def blur(x, s):                            
    k = int(2 * round(3 * s) + 1)
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[float(s), float(s)])


def fog(x, s):                             
    return x * (1 - s) + s * 0.85


def glare(x, s):                           
    _, _, H, W = x.shape
    ys = torch.arange(H, device=x.device).view(1, 1, H, 1).float()
    xs = torch.arange(W, device=x.device).view(1, 1, 1, W).float()
    blob = torch.exp(-(((ys - 0.45 * H) / (0.25 * H)) ** 2 + ((xs - 0.5 * W) / (0.3 * W)) ** 2))
    return x + s * blob


CORRUPTIONS = {
    "low light": (low_light, [0.6, 0.35, 0.15]),
    "blur": (blur, [1, 2, 4]),
    "fog": (fog, [0.3, 0.5, 0.7]),
    "glare": (glare, [0.5, 0.8, 1.2]),
}


def corrupt(img, fn, s):
    x = (img * STD + MEAN)                 
    x = fn(x, s).clamp(0, 1)
    return (x - MEAN) / STD                


@torch.no_grad()
def evaluate(model, loader, fn=None, s=None):
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        if fn is not None:
            img = corrupt(img, fn, s)
        logits = model(pixel_values=img).logits
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        update_confusion(conf, logits.argmax(dim=1), mask)
    iou = iou_from_confusion(conf)
    _, recall = precision_recall_from_confusion(conf)
    return {"miou": iou.mean().item(), "obstacle_iou": iou[0].item(),
            "obstacle_recall": recall[0].item()}


def save_examples(ds, index, path):
    img = ds[index][0].unsqueeze(0).to(device)
    fig, axes = plt.subplots(len(CORRUPTIONS), 4, figsize=(14, 2.8 * len(CORRUPTIONS)))
    for r, (name, (fn, levels)) in enumerate(CORRUPTIONS.items()):
        shown = [("clean", img)] + [(f"{name} {k + 1}", corrupt(img, fn, s))
                                    for k, s in enumerate(levels)]
        for c, (title, x) in enumerate(shown):
            rgb = (x * STD + MEAN).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            axes[r, c].imshow(rgb)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--example", type=int, default=0, help="image index for the example figure")
    args = parser.parse_args()

    ds = MaSTr(f"splits/{args.split}.txt", train=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    model = build_model()
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device, weights_only=True))
    model.to(device).eval()

    results = {"clean": evaluate(model, loader)}
    print(f"clean: {results['clean']}")
    for name, (fn, levels) in CORRUPTIONS.items():
        results[name] = []
        for k, s in enumerate(levels):
            r = evaluate(model, loader, fn, s)
            results[name].append({"severity": k + 1, "value": s, **r})
            print(f"{name:10s} severity {k + 1} (value {s}): mIoU {r['miou']:.4f}  "
                  f"obstacle IoU {r['obstacle_iou']:.4f}  obstacle recall {r['obstacle_recall']:.4f}")

    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results" / f"robustness_{args.split}.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [(axes[0], "miou", "mIoU"), (axes[1], "obstacle_iou", "obstacle IoU")]:
        for name in CORRUPTIONS:
            ys = [results["clean"][key]] + [r[key] for r in results[name]]
            ax.plot(range(4), ys, marker="o", label=name)
        ax.set_xticks(range(4)); ax.set_xticklabels(["clean", "1", "2", "3"])
        ax.set_xlabel("severity"); ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "figures" / f"robustness_{args.split}.png", dpi=110)
    plt.close(fig)

    save_examples(ds, args.example, ROOT / "results" / "figures" / "corruption_examples.png")
    print("saved results/robustness_*.json and results/figures/{robustness_*,corruption_examples}.png")


if __name__ == "__main__":
    main()