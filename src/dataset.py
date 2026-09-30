import random
import shutil
from pathlib import Path


IMG_DIR = Path("./data/raw/images")
MASK_DIR = Path("./data/raw/masks") 
PROCESSED_DIR = Path("./data/processed")

VAL_FRACTION = 0.10
TEST_FRACTION = 0.10
SEED = 42


def match(images_dir: Path, masks_dir: Path) -> list:
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg"}
    )

    pairs = []
    missing = []
    for img_path in image_paths:
        mask_path = masks_dir / f"{img_path.stem}m.png"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            missing.append(img_path.name)

    if missing:
        print(f"WARNING: {len(missing)} images have no matching mask, e.g.: {missing[:5]}")
        print("Check that mask filenames really match image filenames exactly.")

    return pairs


def split_pairs(pairs, val_fraction: float, test_fraction: float, seed: int):
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_val = int(n_total * val_fraction)
    n_test = int(n_total * test_fraction)

    val_pairs = shuffled[:n_val]
    test_pairs = shuffled[n_val:n_val + n_test]
    train_pairs = shuffled[n_val + n_test:]

    return train_pairs, val_pairs, test_pairs


def copy_split(pairs, split_name: str):
    img_out = PROCESSED_DIR / split_name / "images"
    mask_out = PROCESSED_DIR / split_name / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    for img_path, mask_path in pairs:
        shutil.copy2(img_path, img_out / img_path.name)
        shutil.copy2(mask_path, mask_out / mask_path.name)

    print(f"{split_name}: {len(pairs)} pairs -> {img_out} / {mask_out}")


def main():
    if not IMG_DIR.exists() or not MASK_DIR.exists():
        raise SystemExit(
            f"Expected {IMG_DIR} and {MASK_DIR} to exist. "
            "Run download_data.sh first, or check your paths."
        )

    sample_images = sorted(p.name for p in IMG_DIR.iterdir())[:5]
    sample_masks = sorted(p.name for p in MASK_DIR.iterdir())[:5]
    print(f"Sample image filenames: {sample_images}")
    print(f"Sample mask filenames:  {sample_masks}")
    print()

    pairs = match(IMG_DIR, MASK_DIR)
    print(f"Matched {len(pairs)} image/mask pairs total\n")

    train_pairs, val_pairs, test_pairs = split_pairs(pairs, VAL_FRACTION, TEST_FRACTION, SEED)

    copy_split(train_pairs, "train")
    copy_split(val_pairs, "val")
    copy_split(test_pairs, "test")


if __name__ == "__main__":
    main()