import sys
import torch
from PIL import Image
from torchvision import transforms

from model import DisasterCNN

CLASS_NAMES = ["destroyed", "major_damage", "minor_damage", "no_damage"]
MODEL_PATH  = "models/disaster_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = DisasterCNN(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def predict(img_path: str):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = MODEL(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        idx = probs.argmax().item()
        return CLASS_NAMES[idx], probs[idx].item()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict_image.py <path_to_png>")
        sys.exit(1)

    MODEL = load_model()
    cls, conf = predict(sys.argv[1])
    print(f"{cls} ({conf*100:.2f}%)")
