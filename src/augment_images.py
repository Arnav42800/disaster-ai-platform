import os
from PIL import Image
import torchvision.transforms as T
import random

# Configuration
TARGET_DIR = "data/images"
AUGMENT_LIMIT = 1000  # target per class

# Define transformations
augmentations = T.Compose([
    T.RandomRotation(30),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomResizedCrop(64, scale=(0.8, 1.0)),
])

# Augment underrepresented classes
for class_name in ["no-damage", "minor-damage", "major-damage", "destroyed"]:
    class_path = os.path.join(TARGET_DIR, class_name)
    images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    n_existing = len(images)
    n_needed = AUGMENT_LIMIT - n_existing
    print(f"{class_name}: {n_existing} images, need {n_needed} more...")

    if n_needed <= 0:
        continue

    for i in range(n_needed):
        img_name = random.choice(images)
        img_path = os.path.join(class_path, img_name)
        with Image.open(img_path).convert("RGB") as img:
            aug_img = augmentations(img)
            aug_img.save(os.path.join(class_path, f"{os.path.splitext(img_name)[0]}_aug{i}.png"))

print("Augmentation complete.")
