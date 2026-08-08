from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from disaster_ai.config import DEFAULT_MODEL_PATH
from disaster_ai.inference import DamageClassifier


@st.cache_resource
def load_classifier(checkpoint_path: str):
    return DamageClassifier(Path(checkpoint_path))


st.set_page_config(page_title="Disaster Damage Classifier", layout="centered")
st.title("Disaster Damage Classifier")

checkpoint_path = st.sidebar.text_input("Checkpoint", str(DEFAULT_MODEL_PATH))
uploaded = st.file_uploader("Upload a satellite tile", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded tile", use_container_width=True)

    try:
        classifier = load_classifier(checkpoint_path)
        result = classifier.predict_image(image)
    except FileNotFoundError:
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.stop()
    except RuntimeError as exc:
        st.error(f"Could not load checkpoint: {exc}")
        st.stop()

    confidence = result["confidence"] * 100
    st.metric("Prediction", result["predicted_class"], f"{confidence:.2f}% confidence")

    probabilities = pd.DataFrame(
        {
            "class": list(result["probabilities"].keys()),
            "probability": [value * 100 for value in result["probabilities"].values()],
        }
    )
    st.bar_chart(probabilities, x="class", y="probability")

    with st.expander("Model metadata"):
        st.json(classifier.metadata)
