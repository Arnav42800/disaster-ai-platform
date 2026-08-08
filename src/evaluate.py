import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from disaster_ai.config import CLASS_NAMES, CLASS_TO_IDX, DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR, DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_PATH
from disaster_ai.data import ManifestImageDataset, build_manifest, make_transforms
from disaster_ai.metrics import confusion_matrix_frame, classification_report_frame, compute_classification_metrics
from disaster_ai.model import build_model
from disaster_ai.training import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a disaster damage classifier checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.artifacts_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args.data_dir, args.artifacts_dir / "manifest.csv")
    dataset = ManifestImageDataset(
        manifest,
        split=args.split,
        class_to_idx=CLASS_TO_IDX,
        transform=make_transforms(args.image_size, train=False),
    )
    if len(dataset) == 0:
        raise SystemExit(f"No images found for split {args.split!r}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", CLASS_TO_IDX)
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            y_true.extend(labels.tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    classification_report_frame(y_true, y_pred, class_names).to_csv(
        output_dir / "classification_report.csv"
    )
    confusion_matrix_frame(y_true, y_pred, class_names).to_csv(output_dir / "confusion_matrix.csv")

    print(json.dumps(metrics, indent=2))
    print(f"Wrote evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
