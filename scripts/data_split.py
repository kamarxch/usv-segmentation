import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] 

IMG_DIR = ROOT / "data/MaSTr1325/images"
MASK_DIR = ROOT / "data/MaSTr1325/masks"
SPLITS_DIR = ROOT / "splits"
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10
BLOCK = 50
SEED = 42


def match(images_dir, masks_dir):
    pairs = []
    for img in sorted(images_dir.glob("*.jpg")):
        mask = masks_dir / f"{img.stem}m.png"
        assert mask.exists(), f"No mask for {img.name}"
        pairs.append((img, mask))
    assert len(pairs) == 1325, f"Expected 1325 pairs, got {len(pairs)}"
    return pairs


def split_pairs(pairs, val_fraction, test_fraction, seed, block=BLOCK):
    blocks = [pairs[i:i + block] for i in range(0, len(pairs), block)]
    rng = random.Random(seed)
    rng.shuffle(blocks)

    n_val = max(1, round(len(blocks) * val_fraction))
    n_test = max(1, round(len(blocks) * test_fraction))

    flat = lambda bs: [p for b in bs for p in b]
    val_pairs = flat(blocks[:n_val])
    test_pairs = flat(blocks[n_val:n_val + n_test])
    train_pairs = flat(blocks[n_val + n_test:])
    return train_pairs, val_pairs, test_pairs


def write_split(pairs, name):
    SPLITS_DIR.mkdir(exist_ok=True)
    with open(SPLITS_DIR / f"{name}.txt", "w") as f:
        for img, mask in pairs:
            f.write(f"{img.relative_to(ROOT)},{mask.relative_to(ROOT)}\n")
    print(f"{name}: {len(pairs)} pairs")


def main():
    if not IMG_DIR.exists() or not MASK_DIR.exists():
        raise SystemExit(f"Expected {IMG_DIR} and {MASK_DIR} to exist, check your paths.")

    pairs = match(IMG_DIR, MASK_DIR)
    print(f"Matched {len(pairs)} image/mask pairs total\n")

    train, val, test = split_pairs(pairs, VAL_FRACTION, TEST_FRACTION, SEED)

    # sanity check: no image should appear in two splits
    sets = [set(p[0] for p in s) for s in (train, val, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "Splits overlap!"

    write_split(train, "train")
    write_split(val, "val")
    write_split(test, "test")


if __name__ == "__main__":
    main()