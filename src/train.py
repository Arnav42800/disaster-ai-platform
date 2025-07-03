import os
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import DisasterCNN

DATA_DIR  = "data/images"
MODEL_DIR = "models"
MODEL_OUT = os.path.join(MODEL_DIR, "disaster_cnn.pth")

EPOCHS      = 10
BATCH_SIZE  = 32
LR          = 1e-3
VAL_SPLIT   = 0.2
NUM_CLASSES = 4


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Basic transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

    if len(dataset) == 0:
        print("No images found in data/images/* . Run prepare_xview_data.py first.")
        return

    print("Classes & indices:", dataset.class_to_idx)

    # Train/val split
    val_len = int(len(dataset) * VAL_SPLIT)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = DisasterCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total * 100

        # Validation
        model.eval()
        val_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += labels.size(0)

        val_loss /= v_total
        val_acc  = v_correct / v_total * 100

        print(f"Epoch {epoch:02d}: "
              f"Train Loss {train_loss:.4f}  Acc {train_acc:5.2f}% | "
              f"Val Loss {val_loss:.4f}  Acc {val_acc:5.2f}%")

        # Save best
        if val_acc > best_val:
            best_val = val_acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  ✔️  Saved best model ({best_val:.2f}%) → {MODEL_OUT}")


if __name__ == "__main__":
    main()
