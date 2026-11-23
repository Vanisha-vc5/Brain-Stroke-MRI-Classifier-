"""ResNet-18 with the custom 3-class head used in training."""
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from src.config import CLASS_NAMES, MODEL_PATH


def build_resnet18_3class(num_classes: int = 3) -> nn.Module:
    """Same architecture as the training notebook.

    ImageNet weights are not downloaded here: the trained checkpoint already
    contains every backbone weight.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


def _to_float_list(values):
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


def _numpy2_pickle_compat():
    """The checkpoint was pickled with numpy 2 (numpy._core). Let numpy 1.x read it."""
    if int(np.__version__.split(".")[0]) < 2 and "numpy._core" not in sys.modules:
        import numpy.core
        import numpy.core.multiarray
        sys.modules["numpy._core"] = numpy.core
        sys.modules["numpy._core.multiarray"] = numpy.core.multiarray


def load_model(path=MODEL_PATH):
    """Load the trained checkpoint onto the CPU.

    Returns (model, meta) where meta holds class names, validation scores and
    the training history saved alongside the weights.
    """
    # The checkpoint stores numpy scalars next to the state dict, so it needs
    # weights_only=False. It is our own trained file, not untrusted input.
    _numpy2_pickle_compat()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    num_classes = int(ckpt.get("num_classes", len(CLASS_NAMES)))
    model = build_resnet18_3class(num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    history = {}
    for key, values in (ckpt.get("history") or {}).items():
        as_floats = _to_float_list(values)
        if as_floats:
            history[key] = as_floats

    meta = {
        "class_names": list(ckpt.get("class_names") or CLASS_NAMES),
        "val_auc": float(ckpt["val_auc"]) if ckpt.get("val_auc") is not None else None,
        "val_acc": float(ckpt["val_acc"]) if ckpt.get("val_acc") is not None else None,
        "history": history,
    }
    return model, meta
