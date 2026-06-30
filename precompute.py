#!/usr/bin/env python3
"""
Fast precompute using PyTorch multi-threading.
Run this if you want sentence-BERT embeddings.

Usage:
    python precompute.py --candidates data/candidates.jsonl --model models/all-MiniLM-L6-v2
"""

import argparse
import gzip
import json
import time
import os
import numpy as np
from pathlib import Path

def load_candidates(path):
    p = Path(path)
    opener = (lambda: gzip.open(p, "rt", encoding="utf-8")) if p.suffix == ".gz" \
             else (lambda: open(p, "r", encoding="utf-8"))
    candidates = []
    with opener() as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates

def build_compact_text(c):
    p = c["profile"]
    parts = [
        p.get("headline", ""),
        f"{p.get('current_title','')} at {p.get('current_company','')}",
        p.get("summary", "")[:200],
    ]
    for role in c.get("career_history", []):
        if role.get("is_current", False):
            parts.append(role.get("description", "")[:150])
            break
    skills_sorted = sorted(c.get("skills", []),
        key=lambda s: {"expert":4,"advanced":3,"intermediate":2,"beginner":1}.get(
            s.get("proficiency","beginner"), 1), reverse=True)
    parts.append(" ".join(s["name"] for s in skills_sorted[:6]))
    return " ".join(filter(None, parts))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--model",      required=True)
    parser.add_argument("--out",        default="embeddings.npy")
    parser.add_argument("--ids-out",    default="candidate_ids.json")
    args = parser.parse_args()

    # Use ALL CPU cores for torch
    import torch
    num_cores = os.cpu_count()
    torch.set_num_threads(num_cores)
    print(f"Using {num_cores} CPU cores for PyTorch", flush=True)

    from sentence_transformers import SentenceTransformer

    t_start = time.time()

    print("Loading candidates...", flush=True)
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates):,} candidates.", flush=True)

    print(f"Loading model from {args.model}...", flush=True)
    model = SentenceTransformer(args.model)

    print("Building compact texts...", flush=True)
    candidate_ids = [c["candidate_id"] for c in candidates]
    texts = [build_compact_text(c) for c in candidates]
    avg_len = sum(len(t) for t in texts) / len(texts)
    print(f"Avg text length: {avg_len:.0f} chars", flush=True)

    print(f"\nEncoding {len(texts):,} candidates with {num_cores} threads...", flush=True)
    t2 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=512,           # large batch for 12-core machine
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    encode_time = time.time() - t2
    print(f"\nEncoding complete in {encode_time:.1f}s ({encode_time/60:.1f} min)", flush=True)
    print(f"Speed: {len(texts)/encode_time:.0f} candidates/sec", flush=True)

    np.save(args.out, embeddings.astype(np.float32))
    mb = Path(args.out).stat().st_size / 1e6
    print(f"Saved {mb:.1f} MB to {args.out}", flush=True)

    with open(args.ids_out, "w") as f:
        json.dump(candidate_ids, f)

    total = time.time() - t_start
    print(f"\n✓ Done in {total:.1f}s ({total/60:.1f} min)")
    print(f"\nNow run:")
    print(f"  python rank.py --candidates {args.candidates} --embeddings {args.out} --ids {args.ids_out} --out output/team_XUINO.csv")

if __name__ == "__main__":
    main()