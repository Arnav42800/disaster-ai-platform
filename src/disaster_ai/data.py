import re
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from disaster_ai.config import CLASS_NAMES, EVENT_TO_SPLIT, NORMALIZATION

FILENAME_RE = re.compile(r"^(?P<event>.+)_\d{8}_post_disaster\.(png|jpg|jpeg)$", re.IGNORECASE)


def parse_event_from_filename(filename: str) -> str:
    match = FILENAME_RE.match(Path(filename).name)
    if not match:
        raise ValueError(
            "Expected filename like '<event>_<8-digit tile id>_post_disaster.png', "
            f"got {filename!r}"
        )
    return match.group("event")


def split_for_event(event: str) -> str:
    try:
        return EVENT_TO_SPLIT[event]
    except KeyError as exc:
        raise ValueError(f"No fixed split configured for event {event!r}") from exc


def build_manifest(data_dir: Path | str, output_path: Path | str | None = None) -> pd.DataFrame:
    data_root = Path(data_dir)
    rows = []

    for label in CLASS_NAMES:
        class_dir = data_root / label
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.glob("*")):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            event = parse_event_from_filename(image_path.name)
            rows.append(
                {
                    "image_path": str(image_path),
                    "label": label,
                    "event": event,
                    "split": split_for_event(event),
                }
            )

    manifest = pd.DataFrame(rows, columns=["image_path", "label", "event", "split"])
    if not manifest.empty:
        manifest = manifest.sort_values(["split", "label", "event", "image_path"]).reset_index(drop=True)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out, index=False)

    return manifest


def make_transforms(image_size: int, train: bool = False) -> transforms.Compose:
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"]),
        ]
    )
    return transforms.Compose(steps)


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        class_to_idx: dict[str, int],
        transform: Callable | None = None,
    ):
        self.records = manifest[manifest["split"] == split].reset_index(drop=True)
        self.split = split
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image = Image.open(row.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[row.label]
