# train.py
# -*- coding: utf-8 -*-
import argparse, numpy as np, torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from models.base_model import EnhancedMFA_ACP
from utils.base_training import train_one_fold, save_fold_results, save_final_summary
from utils.dataloader import load_full_dataset
import os
import random
os.environ['CUDA_VISIBLE_DEVICES']='0'
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def main():
    parser = argparse.ArgumentParser(description="ACP Training with Cross-Validation")
    parser.add_argument("--dataset", type=str, default="ACP740",
                        choices=["ACPMain", "ACPAlternate", "ACP740", "DeepGram"])
    parser.add_argument("--save_dir", type=str, default="checkpoints/Test/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_wavelet_on_x1", action="store_true")
    parser.add_argument("--use_wavelet_on_x2", action="store_true")
    parser.add_argument("--disable_bigru", action="store_true")
    parser.add_argument("--disable_transformer",default=True,action="store_true")
    parser.add_argument("--disable_film", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5) #Base
    parser.add_argument("--label_smoothing", type=float, default=0.1) #base
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    mod_tags = []
    if args.disable_bigru:        mod_tags.append("noBiGRU")
    if args.disable_transformer:  mod_tags.append("noTR")
    if args.disable_film:         mod_tags.append("noFiLM")
    if args.use_wavelet_on_x1:    mod_tags.append("WVx1")
    if args.use_wavelet_on_x2:    mod_tags.append("WVx2")
    # change parameters
    hp_str = f"lr{args.lr}_wd{args.weight_decay}_sm{args.label_smoothing}_do{args.dropout}_w{args.width}"
    mod_str = "BASE" if len(mod_tags) == 0 else "+".join(mod_tags)


    dataset_layer = args.dataset
    save_dir = os.path.join(args.save_dir, dataset_layer, mod_str, hp_str)
    data_path = f"data/{args.dataset}/train.csv"
    os.makedirs(save_dir, exist_ok=True)

    try:
        import json
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] fail to write config.json: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_full_dataset(data_path, max_len=args.max_len)
    labels = np.array(dataset.get_labels())

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(labels), labels)):
        print(f"\n[Fold {fold}] starting...")
        train_loader = DataLoader(Subset(dataset, tr_idx), batch_size=args.batch_size,
                                  shuffle=True, num_workers=4)
        val_loader = DataLoader(Subset(dataset, va_idx), batch_size=args.batch_size,
                                  shuffle=False, num_workers=4)

        model =EnhancedMFA_ACP(
            x1_channels=20, x2_channels=5, esm_token_dim=1280,
            use_wavelet_on_x1=args.use_wavelet_on_x1,
            use_wavelet_on_x2=args.use_wavelet_on_x2,
            width=args.width,
            use_bigru=not args.disable_bigru,
            use_transformer=not args.disable_transformer,
            use_film=not args.disable_film,
            film_cond_dim=1280,
            max_len=args.max_len,
            dropout=args.dropout
        ).to(device)

        y_true, y_prob, metrics = train_one_fold(
            model, train_loader, val_loader, fold, save_dir, device,
            base_lr=args.lr, weight_decay=args.weight_decay,
            total_epochs=args.epochs, warmup_epochs=args.warmup,
            label_smoothing=args.label_smoothing, grad_clip=args.grad_clip,
            amp=args.amp
        )
        fold_model_path = os.path.join(save_dir, f"model_fold{fold + 1}.pt")
        torch.save(model.state_dict(), fold_model_path)
        print(f"[Saved] Fold {fold + 1} model saved at: {fold_model_path}")

        save_fold_results(metrics, save_dir, fold + 1, y_true, y_prob)
        fold_metrics.append(metrics)

    save_final_summary(fold_metrics, save_dir)

if __name__ == "__main__":
    main()
