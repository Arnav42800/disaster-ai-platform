import json
import sys

import pandas as pd
import pytest
import torch
from PIL import Image

import evaluate
from disaster_ai.inference import DamageClassifier
from disaster_ai.model import build_model
from disaster_ai.training import save_checkpoint
from predict_api import create_app


@pytest.fixture
def evaluation_run(tmp_path, monkeypatch):
    # Training majority differs from test majority: fitting a baseline on test
    # labels would incorrectly report 75% instead of 25% accuracy.
    for event, counts in {
        "hurricane-harvey": {"no_damage": 3, "destroyed": 1},
        "mexico-earthquake": {"no_damage": 1, "destroyed": 3},
    }.items():
        for label, count in counts.items():
            directory = tmp_path / "images" / label
            directory.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                Image.new("RGB", (80, 80), (128, 64, 32)).save(
                    directory / f"{event}_{i:08d}_post_disaster.png"
                )
    model = build_model(model_name="resnet18")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc.bias[0] = 4
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, {
        "model_name": "resnet18",
        "class_to_idx": {"no_damage": 0, "minor_damage": 1, "major_damage": 2, "destroyed": 3},
        "image_size": 32,
        "normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]},
        "train_config": {"split_strategy": "event", "seed": 42},
    })
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--checkpoint", str(checkpoint),
        "--data-dir", str(tmp_path / "images"),
        "--artifacts-dir", str(tmp_path / "results"),
    ])
    evaluate.main()
    return tmp_path, json.loads((tmp_path / "results/test/metrics.json").read_text())


def test_evaluation_uses_checkpoint_class_mapping(evaluation_run):
    _, metrics = evaluation_run
    assert metrics["accuracy"] == 0.25
    assert metrics["per_class"]["no_damage"]["support"] == 1
    assert metrics["per_class"]["destroyed"]["support"] == 3


def test_evaluation_records_preprocessing_and_predictions(evaluation_run):
    root, metrics = evaluation_run
    assert metrics["image_size"] == 32
    assert metrics["normalization"] == {"mean": [0, 0, 0], "std": [1, 1, 1]}
    predictions = pd.read_csv(root / "results/test/predictions.csv")
    assert len(predictions) == 4
    assert set(predictions.predicted_class) == {"no_damage"}
    assert predictions.true_class.value_counts().to_dict() == {"destroyed": 3, "no_damage": 1}
    assert predictions.confidence.tolist() == pytest.approx([0.947915] * 4)


def test_baseline_is_fit_on_training_labels(evaluation_run):
    _, metrics = evaluation_run
    baseline = metrics["majority_baseline"]
    assert baseline["predicted_class"] == "no_damage"
    assert baseline["accuracy"] == 0.25
    assert baseline["macro_f1"] == pytest.approx(0.1)


def test_inference_applies_saved_normalization(evaluation_run):
    root, _ = evaluation_run
    classifier = DamageClassifier(root / "model.pt", device="cpu")
    tensor = classifier.transform(Image.new("RGB", (80, 80), (128, 64, 32)))
    assert tensor.shape == (3, 32, 32)
    assert tensor[:, 0, 0].tolist() == pytest.approx([128 / 255, 64 / 255, 32 / 255])


def test_evaluation_matches_api_for_nondefault_checkpoint(evaluation_run):
    root, _ = evaluation_run
    # Nonconstant outputs make incorrect resizing or normalization observable.
    checkpoint = torch.load(root / "model.pt", weights_only=True)
    torch.manual_seed(7)
    checkpoint["state_dict"] = build_model(model_name="resnet18").state_dict()
    torch.save(checkpoint, root / "model.pt")
    evaluate.main()
    predictions = pd.read_csv(root / "results/test/predictions.csv")
    client = create_app(checkpoint_path=root / "model.pt").test_client()
    with open(predictions.iloc[0].image_path, "rb") as image:
        response = client.post("/predict", data={"image": (image, "tile.png")})
    assert response.status_code == 200
    result = response.get_json()
    assert predictions.iloc[0].predicted_class == result["predicted_class"]
    assert predictions.iloc[0].confidence == pytest.approx(result["confidence"], abs=1e-6)
