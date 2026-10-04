import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from model import build_model

device = "cuda"
model = build_model().to(device).eval()
x = torch.randn(1, 3, 384, 512, device=device)

with torch.no_grad():
    for _ in range(20):                     
        model(pixel_values=x)
    torch.cuda.synchronize()                 
    t = time.perf_counter()
    for _ in range(100):
        model(pixel_values=x)
    torch.cuda.synchronize()
dt = (time.perf_counter() - t) / 100

print(f"{torch.cuda.get_device_name(0)}: {dt * 1000:.1f} ms/frame, {1 / dt:.1f} FPS "
      f"(batch 1, 384x512, fp32, model forward pass only)")