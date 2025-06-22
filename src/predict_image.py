import torch
from torchvision import transforms
from PIL import Image
from image_model import DisasterCNN
import sys

CLASS_NAMES = ['no_damage', 'minor_damage', 'major_damage', 'destroyed']
MODEL_PATH = "models/disaster_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DisasterCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted = torch.argmax(outputs, dim=1).item()
    
    return CLASS_NAMES[predicted]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict_image.py path_to_image.jpg")
    else:
        label = predict(sys.argv[1])
        print(f"Predicted class: {label}")
