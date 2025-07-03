import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from model import DisasterCNN

CLASS_NAMES = ["destroyed", "major_damage", "minor_damage", "no_damage"]
MODEL_PATH  = "models/disaster_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = DisasterCNN(num_classes=len(CLASS_NAMES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Streamlit
st.set_page_config(page_title="Disaster Damage Classifier", layout="centered")
st.title("🌩️ Disaster Damage Classifier")

uploaded = st.file_uploader("Upload a satellite PNG/JPG", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_cls = CLASS_NAMES[pred_idx]
    conf     = probs[pred_idx] * 100

    st.subheader(f"Prediction: **{pred_cls}**  ({conf:.2f}% confidence)")

    st.write("### Class Probabilities")
    for cls, p in zip(CLASS_NAMES, probs):
        st.write(f"- {cls}: {p*100:.2f}%")
