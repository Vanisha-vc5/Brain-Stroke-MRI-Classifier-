"""Precompute similar-case embeddings: python -m src.build_embeddings

The app builds them automatically on first use if this file is missing.
"""
from src.config import EMBEDDINGS_PATH
from src.model import load_model
from src.retrieval import load_or_build_gallery


def main():
    model, _ = load_model()
    gallery = load_or_build_gallery(
        model, progress=lambda f: print(f"\r{f:6.1%}", end="", flush=True)
    )
    print(f"\nSaved {len(gallery['embeddings'])} embeddings to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
