"""Preprocessing, prediction and embedding extraction."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

# Identical to val_transforms in the training notebook.
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def preprocess(image: Image.Image) -> torch.Tensor:
    return TRANSFORM(image.convert("RGB")).unsqueeze(0)


def backbone_features(model, x: torch.Tensor) -> torch.Tensor:
    """512-d global-average-pooled ResNet-18 features (input to the head)."""
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


@torch.no_grad()
def predict(model, image: Image.Image, class_names) -> dict:
    x = preprocess(image)
    feats = backbone_features(model, x)
    logits = model.fc(feats)
    probs = torch.softmax(logits, dim=1)[0].numpy()
    pred_idx = int(np.argmax(probs))
    return {
        "tensor": x,
        "probs": probs,
        "pred_idx": pred_idx,
        "pred_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "embedding": F.normalize(feats, dim=1)[0].numpy().astype(np.float32),
    }


def colour_score(image: Image.Image) -> float:
    """Mean channel spread; near 0 for greyscale scans, large for colour photos."""
    arr = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.float32)
    return float(np.mean(arr.max(axis=2) - arr.min(axis=2)))
