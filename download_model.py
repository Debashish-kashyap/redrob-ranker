#!/usr/bin/env python3
"""
Download sentence-transformers model to local folder.
Run this ONCE with internet connection before running precompute.py.

Usage:
    python download_model.py
"""

import os
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR  = "./models/all-MiniLM-L6-v2"

def main():
    if Path(MODEL_DIR).exists():
        print(f"Model already exists at {MODEL_DIR}")
        return

    print(f"Downloading {MODEL_NAME} (~22MB)...")
    print("This requires internet. Run once, then you're offline-ready.")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    print(f"Saving model to {MODEL_DIR}...")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR)

    print(f"✓ Model saved to {MODEL_DIR}")
    print(f"✓ Size: {sum(f.stat().st_size for f in Path(MODEL_DIR).rglob('*') if f.is_file())/1e6:.1f} MB")
    print()
    print("Next step:")
    print(f"  python precompute.py --candidates data/candidates.jsonl --model {MODEL_DIR}")

if __name__ == "__main__":
    main()