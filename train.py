# -*- coding: utf-8 -*-
"""Train DGMFA using stratified cross-validation.

For each fold, ``train_one_fold`` must select the best epoch using validation
data and save exactly one deployable checkpoint:

    fold{fold}_best_model.pt

The independent test set is not loaded or used during training.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from dataloader import load_full_dataset
from models.base_model_v8_1 import DGMFA
from utils.base_training_v8 import (
    save_final_summary,
    save_fold_results,
    train_one_fold,
)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DGMFA with stratified K-fold cross-validation."
    )
    parser.add_argument(
        "--dataset",
        default="ACPMain",
        choices=("ACPMain", "ACPAlternate", "ACP740", "DeepGram"),
    )
    parser.add_argument("--data_csv", type=Path, default=None)
    parser.add_argument("--save_root", type=Path, default=Path("checkpoints/V8_2"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--early_patience", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gate_init_bias", type=float, default=2.0)
    parser.add_argument("--gate_tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive.")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("--warmup_epochs must be in [0, epochs).")
    if args.early_patience < 1:
        raise ValueError("--early_patience must be positive.")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label_smoothing must be in [0, 1).")
    if args.gate_tau <= 0:
        raise ValueError("--gate_tau must be positive.")


def build_run_directory(args: argparse.Namespace) -> Path:
    """Create a unique, deterministic directory for one training setup."""
    hyperparameter_tag = (
        f"w{args.width}_do{args.dropout}_lr{args.lr}_wd{args.weight_decay}"
        f"_sm{args.label_smoothing}_gb{args.gate_init_bias}_gt{args.gate_tau}"
    )
    run_dir = (
        args.save_root
        / args.dataset
        / "BASE"
        / f"seed{args.seed}"
        / hyperparameter_tag
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_model(args: argparse.Namespace) -> DGMFA:
    """Build the full, non-ablated model used for the main experiment."""
    return DGMFA(
        x1_channels=20,
        x2_channels=5,
        esm_token_dim=1280,
        width=args.width,
        use_bigru=True,
        use_transformer=True,
        use_film=True,
        use_gate=True,
        film_cond_dim=1280,
        max_len=args.max_len,
        dropout=args.dropout,
        gate_init_bias=args.gate_init_bias,
        gate_tau=args.gate_tau,
    )


def verify_fold_artifacts(run_dir: Path, fold: int) -> None:
    """Fail if training did not save the validation-selected checkpoint."""
    checkpoint_path = run_dir / f"fold{fold}_best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"train_one_fold did not create {checkpoint_path}. "
            "The training utility must save the best validation epoch with "
            "this exact filename. Do not substitute the final epoch weights."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_reproducibility(args.seed)

    data_csv = args.data_csv or Path("data") / args.dataset / "train.csv"
    if not data_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found: {data_csv}")

    run_dir = build_run_directory(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        **vars(args),
        "data_csv": str(data_csv),
        "save_root": str(args.save_root),
        "run_dir": str(run_dir),
        "device": str(device),
        "checkpoint_selection_split": "validation",
        "checkpoint_selection_rule": "defined by utils.base_training_v8.train_one_fold",
        "checkpoint_pattern": "fold{fold}_best_model.pt",
        "test_set_used_during_training": False,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2, default=str)

    dataset = load_full_dataset(str(data_csv), max_len=args.max_len)
    labels = np.asarray(dataset.get_labels(), dtype=np.int64)
    if labels.ndim != 1 or len(labels) != len(dataset):
        raise ValueError("Dataset labels must be a one-dimensional array.")

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    fold_metrics = []

    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        print(
            f"\nFold {fold}/{args.folds}: "
            f"train={len(train_indices)}, validation={len(val_indices)}"
        )
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        model = build_model(args).to(device)
        y_true, y_prob, metrics = train_one_fold(
            model,
            train_loader,
            val_loader,
            fold,
            str(run_dir),
            device,
            base_lr=args.lr,
            weight_decay=args.weight_decay,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            amp=args.amp,
            early_patience=args.early_patience,
        )

        # train_one_fold owns checkpoint selection and saving. We deliberately
        # do not save model.state_dict() here: at this point it may contain the
        # final epoch rather than the validation-selected best epoch.
        verify_fold_artifacts(run_dir, fold)
        save_fold_results(metrics, str(run_dir), fold, y_true, y_prob)
        fold_metrics.append(metrics)

    save_final_summary(fold_metrics, str(run_dir))
    print(
        f"\nTraining complete. Saved {args.folds} validation-selected "
        f"checkpoints in: {run_dir}"
    )


if __name__ == "__main__":
    main()
