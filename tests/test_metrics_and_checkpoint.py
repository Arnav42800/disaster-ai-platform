from pathlib import Path

from disaster_ai.config import CLASS_NAMES
from disaster_ai.metrics import compute_classification_metrics
from disaster_ai.model import build_model
from disaster_ai.training import load_checkpoint, save_checkpoint


def test_metrics_include_macro_f1_and_per_class_support():
    result = compute_classification_metrics([0, 1, 1, 2], [0, 1, 2, 2], CLASS_NAMES)

    assert "macro_f1" in result
    assert result["total_examples"] == 4
    assert result["per_class"]["destroyed"]["support"] == 1


def test_checkpoint_round_trip(tmp_path: Path):
    model = build_model(num_classes=4)
    path = tmp_path / "model.pt"
    metadata = {
        "class_to_idx": {name: i for i, name in enumerate(CLASS_NAMES)},
        "image_size": 64,
    }

    save_checkpoint(path, model, metadata)
    loaded = load_checkpoint(path)

    assert loaded["class_to_idx"] == metadata["class_to_idx"]
    assert "state_dict" in loaded


def test_build_model_supports_resnet18():
    model = build_model(num_classes=4, model_name="resnet18")

    assert model.fc.out_features == 4


def test_build_model_accepts_explicit_offline_resnet_weights():
    model = build_model(num_classes=4, model_name="resnet18", pretrained=False)

    assert model.fc.out_features == 4
