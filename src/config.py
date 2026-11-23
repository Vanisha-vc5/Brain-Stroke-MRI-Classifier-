"""Paths and constants shared by the app, pages and offline scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "resnet18_stroke_3class.pth"
EMBEDDINGS_PATH = ROOT / "models" / "gallery_embeddings.npz"
MANIFEST_PATH = ROOT / "dataset" / "final_dataset" / "manifest.csv"
METRICS_PATH = ROOT / "reports" / "metrics.json"
EXAMPLES_DIR = ROOT / "examples"

CLASS_NAMES = ["Normal", "Ischemic", "Haemorrhagic"]
CLASS_COLORS = {
    "Normal": "#2e7d32",
    "Ischemic": "#1565c0",
    "Haemorrhagic": "#c62828",
}

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Similar-case retrieval searches the training split only, so a held-out
# test image is never matched against itself.
GALLERY_SPLIT = "train"
TOP_K_SIMILAR = 5

SOURCE_LABELS = {
    "kaggle_slices": "Kaggle MRI",
    "isles2022": "ISLES-2022",
    "ixi_normal": "IXI",
}

GITHUB_URL = "https://github.com/Vanisha-vc5/Brain-Stroke-MRI-Classifier-"

DISCLAIMER = (
    "Educational and research project only. This model is not clinically "
    "validated and must not be used to diagnose stroke or guide treatment."
)
