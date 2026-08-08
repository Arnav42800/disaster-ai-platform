from io import BytesIO

from PIL import Image

from disaster_ai.config import CLASS_TO_IDX
from disaster_ai.inference import DamageClassifier
from disaster_ai.model import build_model
from disaster_ai.training import save_checkpoint
from predict_api import create_app


class FakeClassifier:
    metadata = {"image_size": 64}

    def predict_image(self, image):
        return {
            "predicted_class": "no_damage",
            "confidence": 0.75,
            "probabilities": {
                "destroyed": 0.05,
                "major_damage": 0.1,
                "minor_damage": 0.1,
                "no_damage": 0.75,
            },
        }


def png_bytes():
    image = Image.new("RGB", (16, 16), color=(20, 40, 60))
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_endpoint():
    app = create_app(FakeClassifier())
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_accepts_image_field():
    app = create_app(FakeClassifier())
    client = app.test_client()

    response = client.post(
        "/predict",
        data={"image": (png_bytes(), "tile.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["predicted_class"] == "no_damage"


def test_predict_accepts_file_field():
    app = create_app(FakeClassifier())
    client = app.test_client()

    response = client.post(
        "/predict",
        data={"file": (png_bytes(), "tile.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["confidence"] == 0.75


def test_damage_classifier_loads_checkpoint_model_type(tmp_path):
    checkpoint = tmp_path / "resnet.pt"
    save_checkpoint(
        checkpoint,
        build_model(model_name="resnet18", pretrained=False),
        {"model_name": "resnet18", "class_to_idx": CLASS_TO_IDX, "image_size": 64},
    )

    classifier = DamageClassifier(checkpoint)

    assert classifier.model.fc.out_features == len(CLASS_TO_IDX)
