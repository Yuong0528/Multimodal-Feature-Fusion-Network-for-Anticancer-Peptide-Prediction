# eval.py
# -*- coding: utf-8 -*-
import os, argparse, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve
)
from models.base_model import EnhancedMFA_ACP_V5
from utils.dataloader import load_test_dataset
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _metrics(y_true, y_prob, thr):
    y_pred = (np.array(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return dict(
        acc=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred),
        specificity=specificity,
        f1=f1_score(y_true, y_pred),
        auc=roc_auc_score(y_true, y_prob),
        mcc=matthews_corrcoef(y_true, y_pred),
        ba=(recall_score(y_true, y_pred) + specificity) / 2,
        gmean=np.sqrt(recall_score(y_true, y_pred) * specificity)
    )


@torch.inference_mode()
def run_one_fold(fold_idx, test_loader, model_kwargs, model_dir):
    model = EnhancedMFA_ACP_V5(**model_kwargs).to(DEVICE)

    ckpt_names = [
        f"fold{fold_idx}_best_model.pt",
        f"best_fold{fold_idx}.pt",
        f"model_fold{fold_idx}.pt",
        f"model_fold{fold_idx+1}.pt"
    ]
    for nm in ckpt_names:
        p = os.path.join(model_dir, nm)
        if os.path.exists(p):
            ckpt = torch.load(p, map_location=DEVICE)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            if any(k.startswith("module.") for k in ckpt.keys()):
                ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt, strict=False)
            break
    else:
        raise FileNotFoundError(f"No model file found for fold {fold_idx} in {model_dir}")

    thr_path = os.path.join(model_dir, f"fold{fold_idx}_best_thr.npy")
    assert os.path.exists(thr_path), f"Missing threshold file: {thr_path}"
    thr = float(np.load(thr_path))

    model.eval()
    ys, ps = [], []
    for batch in test_loader:
        x1, x2, esm_tok, esm_pool, y, mask = batch
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
        esm_tok, esm_pool = esm_tok.to(DEVICE), esm_pool.to(DEVICE)
        mask = mask.to(DEVICE).bool()
        prob = torch.sigmoid(model(x1, x2, esm_tok, esm_pool, mask)).squeeze(1).cpu().numpy()
        ys.extend(y.numpy())
        ps.extend(prob)
    return np.array(ys), np.array(ps), thr


# ===== ;ýp =====
def main():
    parser = argparse.ArgumentParser(description="ACP Evaluation with Cross-Validation")
    parser.add_argument("--dataset", type=str, default="ACPMain",
                        choices=["ACPMain", "ACP740", "DeepGram"])
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_wavelet_on_x1", action="store_true")
    parser.add_argument("--use_wavelet_on_x2", action="store_true")
    parser.add_argument("--disable_bigru", action="store_true")
    parser.add_argument("--disable_transformer",default=True,action="store_true")
    parser.add_argument("--disable_film", action="store_true")
    args = parser.parse_args()

    test_path = f"data/{args.dataset}/test.csv"
    model_dir = os.path.join(args.save_dir, args.dataset)

    result_dir = os.path.join(args.result_dir, args.dataset)
    os.makedirs(result_dir, exist_ok=True)

    test_set = load_test_dataset(test_path, max_len=args.max_len)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model_kwargs = dict(
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
    )

    all_metrics = []
    all_probs = []
    plt.figure(figsize=(10, 8))
    colors = ['#2563eb', '#e11d48', '#8b5cf6', '#059669', '#f59e0b']

    for i in range(args.folds):
        y_true, y_prob, thr = run_one_fold(i, test_loader, model_kwargs, model_dir)
        all_probs.append(y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr,
                 label=f"Fold {i+1} (AUC={roc_auc_score(y_true, y_prob):.3f})",
                 color=colors[i % 5], lw=2.5)
        all_metrics.append(_metrics(y_true, y_prob, thr))

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(result_dir, "per_fold_metrics.csv"), index=False)

    # ===== Mean Probabilities ROC =====
    P = np.stack(all_probs, axis=0).mean(axis=0)
    auc = roc_auc_score(y_true, P)
    fpr, tpr, _ = roc_curve(y_true, P)
    plt.plot(fpr, tpr, label=f"MeanProbs (AUC={auc:.3f})", color="#111827", lw=3.0)
    plt.plot([0, 1], [0, 1], '--', color="#6b7280")
    plt.legend(loc='lower right')
    plt.grid(alpha=0.2)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {args.dataset}")
    plt.savefig(os.path.join(result_dir, "crossval_roc.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ===== G;ß¡ =====
    summary = {k: {"Mean": df[k].mean(), "StdDev": df[k].std()} for k in df.columns}
    sdf = pd.DataFrame(summary).T
    sdf.to_csv(os.path.join(result_dir, "crossval_results.csv"))
    print(f"\n===== {args.dataset} | Final {args.folds}-Fold Test Summary =====")
    print(sdf.to_string(float_format="{:.4f}".format))

    metrics = ["acc", "precision", "recall", "specificity", "f1", "auc", "mcc", "ba", "gmean"]
    means = [df[m].mean() for m in metrics]
    stds = [df[m].std() for m in metrics]

    # colors = [
    #     "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974",
    #     "#64B5CD", "#E17C05", "#937860", "#8C8C8C"
    # ]
    colors = [
             "#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db",
             "#71b7ed", "#b8aeeb", "#f2a7da", "#C3D9B1"
         ]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(
        metrics, means, yerr=stds,
        capsize=6, color=colors, alpha=0.9,
        edgecolor="#1E1E1E", linewidth=1.2
    )

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean:.3f}±{std:.3f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="medium", color="#222"
        )

    ax.set_title(
        f"{args.dataset} | Cross-Validation Performance ({args.folds}-Fold)",
        fontsize=15, fontweight="bold", color="#111"
    )
    ax.set_ylabel("Score", fontsize=12, labelpad=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=11, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for bar in bars:
        bar.set_zorder(3)
        bar.set_alpha(0.92)

    plt.tight_layout()
    bar_path = os.path.join(result_dir, "metrics_barplot.png")
    plt.savefig(bar_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"Finished: {bar_path}")


if __name__ == "__main__":
    main()
