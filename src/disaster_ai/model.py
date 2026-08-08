import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class DisasterCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def build_model(
    num_classes: int = 4,
    model_name: str = "cnn",
    pretrained: bool = False,
) -> nn.Module:
    if model_name == "cnn":
        return DisasterCNN(num_classes=num_classes)
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model {model_name!r}; choose 'cnn' or 'resnet18'")
