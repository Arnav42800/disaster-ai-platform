from io import BytesIO

from PIL import Image

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
