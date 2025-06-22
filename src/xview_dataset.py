# src/xview_dataset.py

import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ["destroyed", "major-damage", "minor-damage", "no-damage"]

class XView2Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for fname in os.listdir(self.label_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(self.label_dir, fname)) as f:
                data = json.load(f)
            features = data.get("features", {}).get("xy", [])
            for feat in features:
                props = feat.get("properties", {})
                damage = props.get("subtype", "").lower()
                if damage in CLASS_NAMES:
                    image_name = fname.replace(".json", ".png")
                    image_path = os.path.join(self.image_dir, image_name)
                    if os.path.exists(image_path):
                        samples.append((image_path, CLASS_NAMES.index(damage)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
