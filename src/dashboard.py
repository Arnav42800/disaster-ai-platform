# src/dashboard.py

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import DisasterCNN

CLASS_NAMES = ["destroyed", "major-damage", "minor-damage", "no-damage"]
MODEL_PATH = "models/disaster_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DisasterCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

st.title("Damage Classifier – xView2")
uploaded_file = st.file_uploader("Upload a satellite image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    predicted = CLASS_NAMES[np.argmax(probs)]
    st.subheader(f"Prediction: `{predicted}`")
    st.write("Confidence scores:")
    for i, cls in enumerate(CLASS_NAMES):
        st.write(f"{cls}: {probs[i] * 100:.2f}%")



