import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from dataset import MaSTr, IGNORE_INDEX
from model import build_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# 10 images, no augmentation (train=False), so the model can memorize them
ds = Subset(MaSTr("splits/train.txt", train=False), range(10))
loader = DataLoader(ds, batch_size=5, shuffle=True)

model = build_model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

for epoch in range(100):
    model.train()
    total = 0.0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)

        logits = model(pixel_values=img).logits      
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)

        loss = loss_fn(logits, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  loss {total / len(loader):.4f}")