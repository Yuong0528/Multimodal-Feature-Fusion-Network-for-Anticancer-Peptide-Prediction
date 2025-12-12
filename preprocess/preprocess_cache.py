# preprocess_cache.py (smart version)
# -*- coding: utf-8 -*-
import os, torch, hashlib, warnings
import pandas as pd
from utils.dataloader import load_esm_per_residue

ESM_CACHE_DIR = "esm_cache_v2"

DATASETS = [
    "data/ACPMain/train.csv",
    "data/ACPMain/test.csv",
    "data/ACPAlternate/train.csv",
    "data/ACPAlternate/test.csv",
    "data/ACP740/train.csv",
    "data/ACP740/test.csv",
    "data/DeepGram/train.csv",
    "data/DeepGram/test.csv"
]

def _seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.strip().upper().encode()).hexdigest()

def main():
    os.makedirs(ESM_CACHE_DIR, exist_ok=True)

    for path in DATASETS:
        if not os.path.exists(path):
            warnings.warn(f"path not exits,pass: {path}")
            continue

        df = pd.read_csv(path)
        seqs = df["sequence"].astype(str).tolist()
        total, done, miss = len(seqs), 0, 0

        print(f"\n>> Processing {path}, {total} sequences")

        for s in seqs:
            h = _seq_hash(s)
            tok_file = os.path.join(ESM_CACHE_DIR, f"{h}.tok.pt")
            pool_file = os.path.join(ESM_CACHE_DIR, f"{h}.pool.pt")

            if os.path.exists(tok_file) and os.path.exists(pool_file):
                done += 1
                continue

            try:
                load_esm_per_residue(s)
                miss += 1
            except Exception as e:
                warnings.warn(f"Failed ESM embedding 1%: {s[:10]}... | {e}")

        print(f"[{path}]has cached: {done}, renew: {miss}, Total: {total}")

if __name__ == "__main__":
    with torch.inference_mode():
        main()
