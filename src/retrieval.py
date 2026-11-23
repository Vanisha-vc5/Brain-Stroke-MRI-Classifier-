"""Similar-case retrieval with cosine similarity on 512-d ResNet-18 embeddings."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import EMBEDDINGS_PATH, GALLERY_SPLIT, MANIFEST_PATH, ROOT
from src.predict import TRANSFORM, backbone_features


def gallery_records() -> pd.DataFrame:
    """Training-split images from the manifest that exist on disk."""
    df = pd.read_csv(MANIFEST_PATH)
    df = df[df["split"] == GALLERY_SPLIT].copy()
    df["abs_path"] = [str(ROOT / p) for p in df["path"]]
    exists = [(ROOT / p).is_file() for p in df["path"]]
    return df[exists].sort_values("path").reset_index(drop=True)


@torch.no_grad()
def compute_embeddings(model, paths, batch_size: int = 32, progress=None) -> np.ndarray:
    chunks = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        batch = torch.stack([TRANSFORM(Image.open(p).convert("RGB")) for p in batch_paths])
        feats = backbone_features(model, batch)
        chunks.append(F.normalize(feats, dim=1).numpy().astype(np.float32))
        if progress is not None:
            progress(min(start + batch_size, len(paths)) / len(paths))
    return np.concatenate(chunks) if chunks else np.zeros((0, 512), dtype=np.float32)


def load_or_build_gallery(model, progress=None) -> dict:
    """Use cached embeddings when they match the current gallery, else rebuild."""
    records = gallery_records()
    paths = records["path"].tolist()

    if EMBEDDINGS_PATH.is_file():
        try:
            cached = np.load(EMBEDDINGS_PATH, allow_pickle=False)
            if cached["paths"].tolist() == paths:
                return {"records": records, "embeddings": cached["embeddings"]}
        except Exception:
            pass

    embeddings = compute_embeddings(model, records["abs_path"].tolist(), progress=progress)
    try:
        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(EMBEDDINGS_PATH, embeddings=embeddings, paths=np.array(paths))
    except OSError:
        pass
    return {"records": records, "embeddings": embeddings}


def most_similar(query: np.ndarray, embeddings: np.ndarray, k: int = 5):
    """Indices and cosine similarities of the k nearest gallery images."""
    if len(embeddings) == 0:
        return []
    sims = embeddings @ query
    k = min(k, len(sims))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(i), float(sims[i])) for i in top]
