# train_v8.py
# -*- coding: utf-8 -*-
"""
Training entrypoint for DGMFA
"""

import os, argparse, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from models.base_model_v8_1 import DGMFA
from utils.base_training_v8 import (
    train_one_fold, save_fold_results, save_final_summary,
)
from dataloader import load_full_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def set_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed);     random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser("ACP Training V8")
    # data / checkpoints
    parser.add_argument("--dataset", type=str, default="ACPMain",
                        choices=["ACPMain", "ACPAlternate", "ACP740", "DeepGram"])
    parser.add_argument("--save_dir", type=str, default="checkpoints/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--width", type=int, default=96)       # your sweet spot
    parser.add_argument("--dropout", type=float, default=0.3)

    # optim
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--early_patience", type=int, default=20)

    # ablations
    parser.add_argument("--disable_bigru",       action="store_true")
    parser.add_argument("--disable_transformer", action="store_true")
    parser.add_argument("--disable_film",        action="store_true")
    parser.add_argument("--disable_gate",        action="store_true",
                        help="When both BiGRU and Transformer are enabled, "
                             "disable the gate and fall back to series "
                             "(V5-style BiGRU -> Transformer).")

    # V8.1 gate tweaks
    parser.add_argument("--gate_init_bias", type=float, default=2.0,
                        help="Initial bias for the gate's last Linear layer. "
                             "+2.0 -> alpha starts ~0.88 (favour BiGRU). "
                             "-2.0 -> alpha starts ~0.12 (favour Transformer). "
                             "0.0 reproduces V8 behaviour (alpha stuck at 0.5).")
    parser.add_argument("--gate_tau", type=float, default=1.0,
                        help="Temperature on the gate logit. tau>1 softens "
                             "(pulls alpha toward 0.5); tau<1 sharpens.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for full reproducibility.")

    args = parser.parse_args()
   # set_seed(42)
    set_seed(args.seed)

    # ---- directory tag ----
    mod_tags = []
    if args.disable_bigru:       mod_tags.append("noGRU")
    if args.disable_transformer: mod_tags.append("noTR")
    if args.disable_film:        mod_tags.append("noFiLM")
    if args.disable_gate and not (args.disable_bigru or args.disable_transformer):
        mod_tags.append("series")
    mod_str = "BASE" if len(mod_tags) == 0 else "+".join(mod_tags)
    hp_str = (f"w{args.width}_do{args.dropout}_lr{args.lr}_sm{args.label_smoothing}"
              f"_gb{args.gate_init_bias}_gt{args.gate_tau}")
    #save_dir = os.path.join(args.save_dir, args.dataset, mod_str, hp_str)
    save_dir = os.path.join(args.save_dir, args.dataset, mod_str,
                            f"seed{args.seed}", hp_str)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"[INFO] save_dir = {save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data ----
    data_path = f"data/{args.dataset}/train.csv"
    dataset = load_full_dataset(data_path, max_len=args.max_len)
    labels = np.array(dataset.get_labels())

    #skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                          random_state=args.seed)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(labels), labels)):
        print(f"\n[Fold {fold + 1}] starting...")
        train_loader = DataLoader(
            Subset(dataset, tr_idx), batch_size=args.batch_size,
            shuffle=True, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(dataset, va_idx), batch_size=args.batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
        )

        model = DGMFA(
            x1_channels=20, x2_channels=5, esm_token_dim=1280,
            width=args.width,
            use_bigru=not args.disable_bigru,
            use_transformer=not args.disable_transformer,
            use_film=not args.disable_film,
            use_gate=not args.disable_gate,
            film_cond_dim=1280,
            max_len=args.max_len,
            dropout=args.dropout,
            gate_init_bias=args.gate_init_bias,
            gate_tau=args.gate_tau,
        ).to(device)

        y_true, y_prob, metrics = train_one_fold(
            model, train_loader, val_loader, fold + 1, save_dir, device,
            base_lr=args.lr, weight_decay=args.weight_decay,
            total_epochs=args.epochs, warmup_epochs=args.warmup,
            label_smoothing=args.label_smoothing, grad_clip=args.grad_clip,
            amp=args.amp, early_patience=args.early_patience,
        )

        fold_model_path = os.path.join(save_dir, f"model_fold{fold + 1}.pt")
        torch.save(model.state_dict(), fold_model_path)

        save_fold_results(metrics, save_dir, fold + 1, y_true, y_prob)
        fold_metrics.append(metrics)

    save_final_summary(fold_metrics, save_dir)
    print(f"\n[Done] Results in: {save_dir}")


if __name__ == "__main__":
    main()

