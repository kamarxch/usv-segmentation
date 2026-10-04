import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from metrics import NUM_CLASSES, update_confusion, iou_from_confusion

from dataset import MaSTr, IGNORE_INDEX, CLASS_NAMES
from model import build_model

ROOT = Path(__file__).resolve().parents[1]
device = "cuda" if torch.cuda.is_available() else "cpu"


def upsample(logits, size):
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
                    

def evaluate(model, loader):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            logits = upsample(model(pixel_values=img).logits, mask.shape[-2:])
            pred = logits.argmax(dim=1)                
            update_confusion(conf, pred, mask)
    return iou_from_confusion(conf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-5)
    args = parser.parse_args()
    print("device:", device, "|", vars(args))

    train_ds = MaSTr("splits/train.txt", train=True)
    val_ds = MaSTr("splits/val.txt", train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    (ROOT / "checkpoints").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    best, history = 0.0, []

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            logits = upsample(model(pixel_values=img).logits, mask.shape[-2:])
            loss = loss_fn(logits, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        iou = evaluate(model, val_loader)
        miou = iou.mean().item()
        train_loss = total / len(train_loader)
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val mIoU {miou:.4f}  " +
              "  ".join(f"{n} {v:.3f}" for n, v in zip(CLASS_NAMES, iou.tolist())))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_miou": miou,
                        "val_iou": dict(zip(CLASS_NAMES, iou.tolist()))})

        if miou > best:
            best = miou
            torch.save(model.state_dict(), ROOT / "checkpoints" / "best.pt")
            print(f"  -> new best ({best:.4f}), saved checkpoints/best.pt")

    with open(ROOT / "results" / "baseline_metrics.json", "w") as f:
        json.dump(history, f, indent=2)
    print("done. best val mIoU:", round(best, 4))


if __name__ == "__main__":
    main()