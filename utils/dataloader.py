# -*- coding: utf-8 -*-
import os, hashlib, warnings
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
Root=os.path.dirname(os.path.abspath(__file__))
Z_SCALE_PATH = os.path.join(Root,'data/zscale.npy')
#Z_SCALE_PATH = 'data/zscale.npy'
ESM_CACHE_DIR = 'esm_cache_v2'
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
os.makedirs(ESM_CACHE_DIR, exist_ok=True)

# }e Z-scale
Z_SCALE = np.load(Z_SCALE_PATH, allow_pickle=True)  # (20,5), float
assert Z_SCALE.shape[0] == 20 and Z_SCALE.shape[1] == 5, "zscale.np: (20,5)"


from models.esm_embed import esm_embed_tokens

def _norm_seq(seq: str) -> str:
    return seq.strip().upper()


def _sha(seq: str) -> str:
    return hashlib.sha256(seq.encode('utf-8')).hexdigest()


def _cache_paths(seq: str):
    h = _sha(_norm_seq(seq))
    tok_path = os.path.join(ESM_CACHE_DIR, f"{h}.tok.pt")
    pool_path = os.path.join(ESM_CACHE_DIR, f"{h}.pool.pt")
    return tok_path, pool_path


@torch.no_grad()
def load_esm_per_residue(seq: str):

    tok_p, pool_p = _cache_paths(seq)
    if os.path.exists(tok_p) and os.path.exists(pool_p):
        tokens = torch.load(tok_p, map_location='cpu')
        pooled = torch.load(pool_p, map_location='cpu')
        return tokens, pooled

    warnings.warn("ESM token cache miss")
    tokens, pooled = esm_embed_tokens([_norm_seq(seq)])

    if isinstance(tokens, tuple) or isinstance(tokens, list):
        tokens, pooled = tokens

    tokens = tokens.detach().cpu().float()   # [L,1280]
    pooled = pooled.detach().cpu().float()   # [1280]
    torch.save(tokens, tok_p)
    torch.save(pooled, pool_p)
    return tokens, pooled


def encode_onehot(seq: str, max_len: int = 50) -> np.ndarray:
    seq = _norm_seq(seq)
    mat = np.zeros((max_len, 20), dtype=np.float32)
    L = min(len(seq), max_len)
    for i in range(L):
        aa = seq[i]
        if aa in AA_TO_INDEX:
            mat[i, AA_TO_INDEX[aa]] = 1.0
    return mat


def encode_zscale(seq: str, max_len: int = 50) -> np.ndarray:
    seq = _norm_seq(seq)
    mat = np.zeros((max_len, 5), dtype=np.float32)
    L = min(len(seq), max_len)
    for i in range(L):
        aa = seq[i]
        if aa in AA_TO_INDEX:
            mat[i] = Z_SCALE[AA_TO_INDEX[aa]]
    return mat

class ACPDataset_Tokens(Dataset):

    def __init__(self, csv_path: str, max_len: int = 50):
        assert os.path.exists(csv_path), f"CSV X(: {csv_path}"
        self.df = pd.read_csv(csv_path)
        if "sequence" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV + 'sequence'  'label' ")

        self.seqs = self.df["sequence"].astype(str).tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        seq = _norm_seq(self.seqs[idx])
        label = float(self.labels[idx])

        x1 = encode_onehot(seq, self.max_len)   # [L,20]
        x2 = encode_zscale(seq, self.max_len)   # [L,5]

        # ESM per-residue + pooled
        tok_raw, pool = load_esm_per_residue(seq)   # tok_raw:[L_raw,1280], pool:[1280]

        D = tok_raw.shape[-1]
        L_raw = tok_raw.shape[0]
        if L_raw >= self.max_len:
            tok = tok_raw[:self.max_len]                          # [max_len, D]
            mask = torch.ones(self.max_len, dtype=torch.bool)     # True=valid
        else:
            pad = torch.zeros(self.max_len - L_raw, D, dtype=tok_raw.dtype)
            tok = torch.cat([tok_raw, pad], dim=0)                # [max_len, D]
            mask = torch.zeros(self.max_len, dtype=torch.bool)
            mask[:L_raw] = True

        x1 = torch.tensor(x1, dtype=torch.float32).transpose(0, 1)
        x2 = torch.tensor(x2, dtype=torch.float32).transpose(0, 1)
        tok = tok.float()
        pool = pool.float()
        y = torch.tensor(label, dtype=torch.float32)

        return x1, x2, tok, pool, y, mask


def load_full_dataset(csv_path: str, max_len: int = 50) -> ACPDataset_Tokens:
    return ACPDataset_Tokens(csv_path, max_len=max_len)



def load_test_dataset(csv_path, max_len=50):
    return ACPDataset_Tokens(csv_path, max_len)