from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

from disaster_ai.config import DEFAULT_MODEL_PATH
from disaster_ai.inference import DamageClassifier


def create_app(classifier=None, checkpoint_path: Path | str = DEFAULT_MODEL_PATH) -> Flask:
    app = Flask(__name__)
    app.config["classifier"] = classifier
    app.config["checkpoint_path"] = Path(checkpoint_path)

    def get_classifier():
        if app.config["classifier"] is None:
            app.config["classifier"] = DamageClassifier(app.config["checkpoint_path"])
        return app.config["classifier"]

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/predict")
    def predict():
        upload = request.files.get("image") or request.files.get("file")
        if upload is None:
            return jsonify({"error": "Upload an image using form field 'image' or 'file'."}), 400

        try:
            image = Image.open(upload.stream).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return jsonify({"error": "Uploaded file is not a readable image."}), 400

        return jsonify(get_classifier().predict_image(image))

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
