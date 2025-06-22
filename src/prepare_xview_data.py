# src/prepare_xview_data.py

import os
import json
import shutil
from PIL import Image
from tqdm import tqdm

# Paths
XVIEW_DIRS = [
    os.path.expanduser("~/Downloads/train"),
    os.path.expanduser("~/Downloads/tier3"),
]
OUTPUT_DIR = "data/images"
DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
CLASS_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}
os.makedirs(OUTPUT_DIR, exist_ok=True)
for cls in DAMAGE_CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

def parse_and_copy_images():
    total_count = 0
    valid_count = 0

    for xview_dir in XVIEW_DIRS:
        images_dir = os.path.join(xview_dir, "images")
        labels_dir = os.path.join(xview_dir, "labels")

        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith(".png") and "_post_disaster" in f
        ])

        for img_file in tqdm(image_files, desc=f"Parsing {xview_dir}"):
            img_path = os.path.join(images_dir, img_file)
            label_file = img_file.replace(".png", ".json")
            label_path = os.path.join(labels_dir, label_file)

            if not os.path.exists(label_path):
                continue

            with open(label_path) as f:
                try:
                    label_data = json.load(f)
                    features = label_data.get("features", [])
                    if not features:
                        continue
                except json.JSONDecodeError:
                    continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            for i, feature in enumerate(features):
                if not isinstance(feature, dict):
                    continue
                props = feature.get("properties", {})
                subtype = props.get("subtype", "").strip().lower()
                if subtype not in DAMAGE_CLASSES:
                    continue

                geom = feature.get("wkt", "")
                if not geom.startswith("POLYGON"):
                    continue

                # Convert WKT polygon string to bounding box
                coords = geom.replace("POLYGON ((", "").replace("))", "").split(", ")
                points = [tuple(map(float, pt.split())) for pt in coords]
                xs, ys = zip(*points)
                xmin, xmax = int(min(xs)), int(max(xs))
                ymin, ymax = int(min(ys)), int(max(ys))

                # Crop and save patch
                crop = image.crop((xmin, ymin, xmax, ymax))
                save_path = os.path.join(OUTPUT_DIR, subtype, f"{img_file[:-4]}_{i}.png")
                try:
                    crop.save(save_path)
                    valid_count += 1
                except Exception:
                    continue

            total_count += 1

    print(f"\n✅ Finished. Parsed {total_count} images. Saved {valid_count} valid crops.")

if __name__ == "__main__":
    parse_and_copy_images()
