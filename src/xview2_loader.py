import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

# Label mapping
LABEL_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

class XView2Dataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.samples = []
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        for root_dir in root_dirs:
            image_dir = os.path.join(root_dir, "images")
            label_dir = os.path.join(root_dir, "labels")

            for fname in os.listdir(image_dir):
                if "_post_disaster" not in fname or not fname.endswith(".png"):
                    continue  # skip pre-disaster or non-PNGs

                image_path = os.path.join(image_dir, fname)
                label_fname = fname.replace(".png", ".json")
                label_path = os.path.join(label_dir, label_fname)

                if not os.path.exists(label_path):
                    continue

                try:
                    with open(label_path, "r") as f:
                        label_json = json.load(f)
                    damage = label_json["features"]["metadata"]["damage"]

                    if damage not in LABEL_MAP:
                        continue

                    label = LABEL_MAP[damage]
                    self.samples.append((image_path, label))

                except Exception as e:
                    print(f"Skipping {label_path}: {e}")

        print(f"Loaded {len(self.samples)} valid post-disaster images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label)
