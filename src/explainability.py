"""Grad-CAM on the last ResNet-18 block (layer4[-1])."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def grad_cam(model, x: torch.Tensor, target_class: int) -> np.ndarray:
    """Return a 7x7-resolution Grad-CAM map in [0, 1] for target_class."""
    store = {}

    def forward_hook(_module, _inputs, output):
        store["activations"] = output
        output.register_hook(lambda grad: store.__setitem__("gradients", grad))

    handle = model.layer4[-1].register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x)
            logits[0, target_class].backward()
    finally:
        handle.remove()

    activations = store["activations"][0].detach()
    gradients = store["gradients"][0].detach()
    weights = gradients.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))
    if float(cam.max()) > 0:
        cam = cam / cam.max()
    model.zero_grad(set_to_none=True)
    return cam.numpy()


def _jet(values: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(image: Image.Image, cam: np.ndarray, strength: float = 0.6) -> Image.Image:
    """Blend the CAM over the original image; cool areas keep the scan visible."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w = base.shape[:2]
    cam_t = torch.from_numpy(cam)[None, None].float()
    cam_up = F.interpolate(cam_t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    cam_up = np.clip(cam_up, 0, 1)
    alpha = (strength * cam_up)[..., None]
    blended = base * (1 - alpha) + _jet(cam_up) * alpha
    return Image.fromarray((blended * 255).astype(np.uint8))
