import os

image_dir = 'data/images'
class_counts = {}

for class_name in os.listdir(image_dir):
    class_path = os.path.join(image_dir, class_name)
    if os.path.isdir(class_path):
        count = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        class_counts[class_name] = count

print("Image counts per class:")
for k, v in class_counts.items():
    print(f"{k}: {v}")