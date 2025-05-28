import os
import cv2
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

ORIG_IMG_DIR = Path("images")
ORIG_LBL_DIR = Path("labels")

TILE_SIZE = 640
OVERLAP = 0.2
VAL_SPLIT = 0.3

TILED_IMG_DIR = Path("dataset/images")
TILED_LBL_DIR = Path("dataset/labels")

def tile_image_and_labels(image_path, label_path):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    step = int(TILE_SIZE * (1 - OVERLAP))

    base_name = image_path.stem
    with open(label_path, 'r') as f:
        labels = [list(map(float, line.strip().split())) for line in f]

    count = 0
    for y in range(0, h - TILE_SIZE + 1, step):
        for x in range(0, w - TILE_SIZE + 1, step):
            tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
            tile_name = f"{base_name}_{x}_{y}.jpg"
            cv2.imwrite(TILED_IMG_DIR / tile_name, tile)

            tile_labels = []
            for cls, cx, cy, bw, bh in labels:
                abs_x, abs_y = cx * w, cy * h
                abs_w, abs_h = bw * w, bh * h
                box_left = abs_x - abs_w / 2
                box_top = abs_y - abs_h / 2

                if (x <= abs_x <= x + TILE_SIZE) and (y <= abs_y <= y + TILE_SIZE):
                    new_cx = (abs_x - x) / TILE_SIZE
                    new_cy = (abs_y - y) / TILE_SIZE
                    new_w = abs_w / TILE_SIZE
                    new_h = abs_h / TILE_SIZE

                    if 0 <= new_cx <= 1 and 0 <= new_cy <= 1:
                        tile_labels.append(f"{int(cls)} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")

            label_file = (TILED_LBL_DIR / tile_name).with_suffix(".txt")
            with open(label_file, 'w') as f:
                f.write('\n'.join(tile_labels))
            count += 1
    print(f"Tiled {count} patches from {image_path.name}")

def main():
    shutil.rmtree("dataset", ignore_errors=True)
    TILED_IMG_DIR.mkdir(parents=True)
    TILED_LBL_DIR.mkdir(parents=True)

    for img_file in ORIG_IMG_DIR.glob("*.jpg"):
        lbl_file = ORIG_LBL_DIR / f"{img_file.stem}.txt"
        if lbl_file.exists():
            tile_image_and_labels(img_file, lbl_file)

    all_tiles = list(TILED_IMG_DIR.glob("*.jpg"))
    train_imgs, val_imgs = train_test_split(all_tiles, test_size=VAL_SPLIT, random_state=42)

    for split in ['train', 'val']:
        Path(f"split/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"split/labels/{split}").mkdir(parents=True, exist_ok=True)

    for split, files in zip(['train', 'val'], [train_imgs, val_imgs]):
        for f in files:
            name = f.name
            shutil.copy(f, f"split/images/{split}/{name}")
            shutil.copy(TILED_LBL_DIR / f"{f.stem}.txt", f"split/labels/{split}/{f.stem}.txt")

    with open("dataset.yaml", "w") as f:
        f.write("""train: split/images/train
val: split/images/val

nc: 2
names: ['door', 'window']
""")
    print("✅ Dataset prepared and split")

if __name__ == "__main__":
    main()
