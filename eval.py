# -*- coding: utf-8 -*-
"""Evaluate trained ACP models on a fixed independent test set.

The script loads one checkpoint per cross-validation fold. Each checkpoint and
its optional decision threshold must have been selected using validation data.
The test set is evaluated exactly once; it is never used to tune a threshold.
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataloader import load_test_dataset
from models.base_model import DGMFA

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_checkpoint(checkpoint_dir, fold):
    """Return the model checkpoint and optional validation threshold."""
    model_names = (
        f"fold{fold}_best_model.pt",
        f"best_fold{fold}.pt",
        f"model_fold{fold}.pt",
    )
    checkpoint_path = next(
        (
            os.path.join(checkpoint_dir, name)
            for name in model_names
            if os.path.isfile(os.path.join(checkpoint_dir, name))
        ),
        None,
    )
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found for fold {fold} in {checkpoint_dir}. "
            f"Expected one of: {', '.join(model_names)}"
        )

    threshold_path = os.path.join(checkpoint_dir, f"fold{fold}_best_thr.npy")
    if not os.path.isfile(threshold_path):
        threshold_path = None
    return checkpoint_path, threshold_path


def load_model(checkpoint_path, model_kwargs, device):
    model = DGMFA(**model_kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def predict(model, data_loader, device):
    labels = []
    probabilities = []

    for x1, x2, esm_token, esm_pool, y, mask in data_loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        esm_token = esm_token.to(device, non_blocking=True)
        esm_pool = esm_pool.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True).bool()

        logits = model(x1, x2, esm_token, esm_pool, mask)
        probability = torch.sigmoid(logits).reshape(-1)

        labels.append(y.reshape(-1).cpu().numpy())
        probabilities.append(probability.cpu().numpy())

    return np.concatenate(labels).astype(int), np.concatenate(probabilities)


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed fold checkpoints on an independent test set."
    )
    parser.add_argument(
        "--dataset",
        default="ACPMain",
        choices=["ACPMain", "ACPAlternate", "ACP740", "DeepGram"],
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing fold{i}_best_model.pt files.",
    )
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--output_dir", default="results/eval")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)

    # These options must exactly match the architecture used for training.
    parser.add_argument("--disable_bigru", action="store_true")
    parser.add_argument("--disable_transformer", action="store_true")
    parser.add_argument("--disable_film", action="store_true")
    parser.add_argument("--disable_gate", action="store_true")
    parser.add_argument("--disable_bpf", action="store_true")
    parser.add_argument("--disable_zscale", action="store_true")
    parser.add_argument("--disable_esm", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.disable_bpf and args.disable_zscale and args.disable_esm:
        raise ValueError("At least one input modality must remain enabled.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_csv = args.test_csv or os.path.join("data", args.dataset, "test.csv")

    test_dataset = load_test_dataset(test_csv, max_len=args.max_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_kwargs = {
        "x1_channels": 20,
        "x2_channels": 5,
        "esm_token_dim": 1280,
        "width": args.width,
        "use_bigru": not args.disable_bigru,
        "use_transformer": not args.disable_transformer,
        "use_film": not args.disable_film,
        "use_gate": not args.disable_gate,
        "use_bpf": not args.disable_bpf,
        "use_zscale": not args.disable_zscale,
        "use_esm": not args.disable_esm,
        "film_cond_dim": 1280,
        "max_len": args.max_len,
        "dropout": args.dropout,
    }

    rows = []
    reference_labels = None
    for fold in range(1, args.folds + 1):
        checkpoint_path, threshold_path = find_checkpoint(args.checkpoint_dir, fold)
        threshold = float(np.load(threshold_path)) if threshold_path else 0.5

        model = load_model(checkpoint_path, model_kwargs, device)
        y_true, y_prob = predict(model, test_loader, device)

        if reference_labels is None:
            reference_labels = y_true
        elif not np.array_equal(reference_labels, y_true):
            raise RuntimeError("Test labels changed between fold evaluations.")

        metrics = compute_metrics(y_true, y_prob, threshold)
        rows.append(
            {
                "fold": fold,
                "checkpoint": checkpoint_path,
                **metrics,
            }
        )
        print(
            f"Fold {fold}: ACC={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1']:.4f}, MCC={metrics['mcc']:.4f}, "
            f"AUC={metrics['auc']:.4f}, threshold={threshold:.4f}"
        )

    results = pd.DataFrame(rows)
    metric_columns = [
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "mcc",
        "auc",
    ]
    summary = results[metric_columns].agg(["mean", "std"]).T

    os.makedirs(args.output_dir, exist_ok=True)
    results.to_csv(os.path.join(args.output_dir, "per_fold_metrics.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "summary_metrics.csv"))
    with open(os.path.join(args.output_dir, "eval_config.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)

    print("\nMean ± SD across folds")
    for metric in metric_columns:
        print(
            f"{metric:>11}: {summary.loc[metric, 'mean']:.4f} "
            f"± {summary.loc[metric, 'std']:.4f}"
        )
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
