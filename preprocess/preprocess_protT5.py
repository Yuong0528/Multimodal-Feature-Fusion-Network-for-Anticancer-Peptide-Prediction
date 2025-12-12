
# preprocess_protT5_gpu.py
import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel

# ---------------- Settings ----------------
MODEL_PATH_T5 = "../data/Prot_T5"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# ---------------- Load ProtT5 ----------------
tokenizer_t5 = T5Tokenizer.from_pretrained(MODEL_PATH_T5, local_files_only=True, use_fast=True)
model_t5 = T5EncoderModel.from_pretrained(MODEL_PATH_T5, local_files_only=True).eval().to(DEVICE)
print(f"[ProtT5] Loaded encoder-only model on {DEVICE}.")

# ---------------- Dataset ----------------
class SeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].strip().upper()
        return seq, self.labels[idx]

def collate_fn(batch, max_len):
    seqs, labels = zip(*batch)
    seqs = [" ".join(list(s)) for s in seqs]
    inputs = tokenizer_t5(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len + 2
    )
    labels = torch.tensor(labels, dtype=torch.float32)
    return inputs, labels

# ---------------- Encode Batch ----------------
def encode_batch(inputs, max_len):
    for k in inputs:
        inputs[k] = inputs[k].to(DEVICE)
    with torch.no_grad():
        out = model_t5(**inputs)
        hidden = out.last_hidden_state  # [B, L, hidden_dim]

    eos_id = tokenizer_t5.eos_token_id
    mask = (inputs['input_ids'] != eos_id) & inputs['attention_mask'].bool()

    batch_features = []
    for i in range(hidden.size(0)):
        h = hidden[i][mask[i]]
        L_eff = min(h.size(0), max_len)
        feat = torch.zeros((max_len, hidden.size(-1)), device=DEVICE)
        if L_eff > 0:
            feat[:L_eff] = h[:L_eff]
        batch_features.append(feat)

    return torch.stack(batch_features)  # [B, max_len, hidden_dim]

# ---------------- Main Preprocess ----------------
def preprocess_and_save(csv_path, save_path, max_len=50):
    df = pd.read_csv(csv_path)
    if 'sequence' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'sequence' and 'label'.")

    dataset = SeqDataset(df['sequence'].tolist(), df['label'].tolist())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, max_len))

    all_features = []
    all_labels = []

    for batch_inputs, batch_labels in tqdm(loader, desc="Encoding batches"):
        feats = encode_batch(batch_inputs, max_len)
        all_features.append(feats.cpu())
        all_labels.append(batch_labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "features": features,
        "labels": labels,
        "max_len": max_len
    }, save_path)

    print(f"[Done] Saved ProtT5 embeddings to {save_path}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute ProtT5 embeddings (GPU batch) and save to .pt")
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV (with 'sequence' and 'label')")
    parser.add_argument("--save", type=str, required=True, help="Path to save the .pt file")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum sequence length (default=50)")
    args = parser.parse_args()

    preprocess_and_save(args.csv, args.save, args.max_len)

