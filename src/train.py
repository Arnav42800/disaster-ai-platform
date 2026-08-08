import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from disaster_ai.config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_PATH,
    DEFAULT_SEED,
    IDX_TO_CLASS,
    NORMALIZATION,
)
from disaster_ai.data import ManifestImageDataset, build_manifest, make_transforms
from disaster_ai.model import build_model
from disaster_ai.training import evaluate_model, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the disaster damage tile classifier.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def class_weights_for(manifest) -> torch.Tensor:
    train_labels = manifest[manifest["split"] == "train"]["label"]
    counts = Counter(train_labels)
    total = sum(counts.values())
    return torch.tensor(
        [total / max(counts.get(label, 0), 1) for label in CLASS_NAMES],
        dtype=torch.float32,
    )


def serializable_config(args: argparse.Namespace, device: torch.device) -> dict:
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    config["device"] = str(device)
    return config


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.artifacts_dir / "manifest.csv"
    manifest = build_manifest(args.data_dir, manifest_path)
    if manifest.empty:
        raise SystemExit(f"No images found under {args.data_dir}. Expected data/images/<class>/*.png")

    train_ds = ManifestImageDataset(
        manifest,
        split="train",
        class_to_idx=CLASS_TO_IDX,
        transform=make_transforms(args.image_size, train=True),
    )
    val_ds = ManifestImageDataset(
        manifest,
        split="val",
        class_to_idx=CLASS_TO_IDX,
        transform=make_transforms(args.image_size, train=False),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    weights = class_weights_for(manifest).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_macro_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.size(0))
            total_examples += int(labels.size(0))

        train_loss = total_loss / total_examples
        val_metrics, _, _, val_loss = evaluate_model(
            model, val_loader, device, CLASS_NAMES, criterion=criterion
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2))

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                args.output,
                model,
                {
                    "model_name": "DisasterCNN",
                    "class_to_idx": dict(CLASS_TO_IDX),
                    "idx_to_class": dict(IDX_TO_CLASS),
                    "image_size": args.image_size,
                    "normalization": NORMALIZATION,
                    "seed": args.seed,
                    "best_epoch": epoch,
                    "val_metrics": val_metrics,
                    "train_config": serializable_config(args, device),
                },
            )
            print(f"Saved new best checkpoint to {args.output}")

    (args.artifacts_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
