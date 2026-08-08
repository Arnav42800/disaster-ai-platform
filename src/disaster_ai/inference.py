from pathlib import Path
from typing import Any

import torch
from PIL import Image

from disaster_ai.config import CLASS_NAMES, CLASS_TO_IDX, DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_PATH
from disaster_ai.data import make_transforms
from disaster_ai.model import build_model
from disaster_ai.training import load_checkpoint


class DamageClassifier:
    def __init__(self, checkpoint_path: Path | str = DEFAULT_MODEL_PATH, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = load_checkpoint(self.checkpoint_path, map_location=self.device)

        self.class_to_idx = checkpoint.get("class_to_idx", dict(CLASS_TO_IDX))
        self.idx_to_class = {
            int(index): name for name, index in self.class_to_idx.items()
        }
        self.class_names = [
            self.idx_to_class[index] for index in sorted(self.idx_to_class)
        ]
        self.image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
        self.metadata: dict[str, Any] = {
            key: value
            for key, value in checkpoint.items()
            if key != "state_dict"
        }
        self.metadata.setdefault("image_size", self.image_size)
        self.metadata.setdefault("class_to_idx", self.class_to_idx)

        model_name = checkpoint.get("model_name", "cnn")
        if model_name == "DisasterCNN":
            model_name = "cnn"
        self.model = build_model(
            num_classes=len(self.class_names),
            model_name=model_name,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.transform = make_transforms(self.image_size, train=False)

    def predict_image(self, image: Image.Image) -> dict[str, Any]:
        rgb_image = image.convert("RGB")
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu()

        pred_idx = int(torch.argmax(probabilities).item())
        probability_map = {
            self.class_names[index]: float(probabilities[index].item())
            for index in range(len(self.class_names))
        }
        return {
            "predicted_class": self.class_names[pred_idx],
            "confidence": probability_map[self.class_names[pred_idx]],
            "probabilities": probability_map,
        }
