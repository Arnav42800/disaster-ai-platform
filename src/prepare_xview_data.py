# src/copy_xview_tiles.py
"""
Copy full 1024×1024 post-disaster tiles from xBD into
data/images/<class>/ based on the majority subtype in the label JSON.

Usage:
    python src/copy_xview_tiles.py
"""

import os
import json
import shutil
from glob import glob
from collections import Counter
from pathlib import Path
from tqdm import tqdm

XBD_ROOT = Path.home() / "Downloads" / "train"
LABEL_DIR = XBD_ROOT / "labels"
IMG_ROOT  = XBD_ROOT / "images"
DEST_ROOT = Path("data/images") 

CLASS_MAP = {
    "no-damage"   : "no_damage",
    "minor-damage": "minor_damage",
    "major-damage": "major_damage",
    "destroyed"   : "destroyed",
}

def ensure_dest_dirs() -> None:
    for cls in CLASS_MAP.values():
        (DEST_ROOT / cls).mkdir(parents=True, exist_ok=True)

def majority_damage(label_path: Path) -> str | None:
    """Return 'destroyed' | 'major-damage' | …   or None if JSON is empty."""
    with label_path.open() as f:
        data = json.load(f)

    # xBD’s JSONs store polygons under features["lng_lat"]  (NOT ["xy"])
    feats = data.get("features", {}).get("lng_lat", [])
    counter = Counter(feat.get("properties", {}).get("subtype", "")
                      for feat in feats)

    # Remove unknown keys, keep only the four we care about
    counter = {k: counter.get(k, 0) for k in CLASS_MAP.keys()}
    if all(v == 0 for v in counter.values()):
        return None
    return max(counter, key=counter.get)

def find_post_png(basename: str) -> Path | None:
    """
    Given 'guatemala-volcano_00000042_post_disaster.png' find that file
    somewhere under …/images/.  (There are sub-folders 'post' and 'pre'.)
    """
    matches = glob(str(IMG_ROOT / "**" / basename), recursive=True)
    return Path(matches[0]) if matches else None

def main() -> None:
    ensure_dest_dirs()

    label_files = sorted(LABEL_DIR.glob("*_post_disaster.json"))
    if not label_files:
        print(f"No JSONs found in {LABEL_DIR}")
        return

    copied = skipped = 0
    for lbl in tqdm(label_files, desc=f"Scanning {LABEL_DIR}"):
        dmg = majority_damage(lbl)
        if dmg is None:
            skipped += 1
            continue

        png_name = lbl.name.replace(".json", ".png")
        src_png  = find_post_png(png_name)
        if not src_png:
            skipped += 1
            continue

        dst = DEST_ROOT / CLASS_MAP[dmg] / png_name
        shutil.copy2(src_png, dst)
        copied += 1

    print(f"\nFinished.  Copied {copied:,} images  •  Skipped {skipped:,}.")

if __name__ == "__main__":
    main()
