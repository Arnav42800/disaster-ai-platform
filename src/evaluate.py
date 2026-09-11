import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from disaster_ai.config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH
from disaster_ai.data import ManifestImageDataset, build_manifest, make_transforms
from disaster_ai.inference import DamageClassifier
from disaster_ai.metrics import confusion_matrix_frame, classification_report_frame, compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a disaster damage classifier checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--split-strategy",
        choices=["event", "stratified"],
        default=None,
        help="defaults to the strategy saved in the checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=None,
                        help="override the image size saved in the checkpoint")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.artifacts_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier = DamageClassifier(args.checkpoint, device=str(device))
    train_config = classifier.metadata.get("train_config", {})
    image_size = classifier.image_size if args.image_size is None else args.image_size
    if image_size <= 0:
        raise SystemExit("--image-size must be positive")
    split_strategy = args.split_strategy or train_config.get("split_strategy", "event")
    manifest = build_manifest(
        args.data_dir,
        args.artifacts_dir / "manifest.csv",
        split_strategy=split_strategy,
        seed=int(train_config.get("seed", 42)),
    )
    dataset = ManifestImageDataset(
        manifest,
        split=args.split,
        class_to_idx=classifier.class_to_idx,
        transform=make_transforms(image_size, normalization=classifier.normalization),
    )
    if len(dataset) == 0:
        raise SystemExit(f"No images found for split {args.split!r}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    class_names = classifier.class_names
    model_name = classifier.metadata.get("model_name", train_config.get("model", "cnn"))
    train_counts = manifest.loc[manifest["split"] == "train", "label"].value_counts()
    if train_counts.empty:
        raise SystemExit("No training labels found for the majority-class baseline")
    majority_class = max(class_names, key=lambda name: int(train_counts.get(name, 0)))

    y_true = []
    y_pred = []
    confidences = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = classifier.model(images)
            y_true.extend(labels.tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            confidences.extend(outputs.softmax(dim=1).max(dim=1).values.cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    metrics["split"] = args.split
    metrics["split_strategy"] = split_strategy
    metrics["model_name"] = model_name
    metrics["image_size"] = image_size
    metrics["normalization"] = classifier.normalization
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["seed"] = int(train_config.get("seed", 42))
    baseline = compute_classification_metrics(
        y_true, [classifier.class_to_idx[majority_class]] * len(y_true), class_names,
    )
    baseline["predicted_class"] = majority_class
    metrics["majority_baseline"] = baseline
    metrics["improvement_over_baseline"] = {
        key: metrics[key] - baseline[key]
        for key in ("accuracy", "balanced_accuracy", "macro_f1")
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    classification_report_frame(y_true, y_pred, class_names).to_csv(
        output_dir / "classification_report.csv"
    )
    confusion_matrix_frame(y_true, y_pred, class_names).to_csv(output_dir / "confusion_matrix.csv")
    predictions = dataset.records[["image_path", "event", "label"]].rename(columns={"label": "true_class"})
    predictions["predicted_class"] = [class_names[index] for index in y_pred]
    predictions["confidence"] = confidences
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
