from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from disaster_ai.metrics import compute_classification_metrics


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path | str, model: nn.Module, metadata: dict[str, Any]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload["state_dict"] = model.state_dict()
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    return {"state_dict": checkpoint}


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str] | tuple[str, ...],
    criterion: nn.Module | None = None,
) -> tuple[dict, list[int], list[int], float | None]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += float(loss.item()) * int(labels.size(0))
                total_examples += int(labels.size(0))
            predictions = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    avg_loss = total_loss / total_examples if criterion is not None and total_examples else None
    return metrics, y_true, y_pred, avg_loss
