import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import DisasterCNN
import os

def get_data_loaders(data_dir="data/images", batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_map = dataset.class_to_idx
    print("Class map:", class_map)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, class_map

def train():
    train_loader, val_loader, _ = get_data_loaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DisasterCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Accuracy: {train_acc:.2f}%")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/disaster_cnn.pth")
    print("Model saved to models/disaster_cnn.pth")

if __name__ == "__main__":
    train()

