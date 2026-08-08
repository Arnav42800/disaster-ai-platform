from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class XViewFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        self.samples = list(self.root.rglob("*.png"))
        self.classes = sorted({p.parent.name for p in self.samples})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.class_to_idx[path.parent.name]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
