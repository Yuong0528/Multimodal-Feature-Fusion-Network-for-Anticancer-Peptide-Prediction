# eval_v8.py
# -*- coding: utf-8 -*-
"""
Evaluation script for DGMFA on an independent test set.
"""
import os, argparse, random, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
    balanced_accuracy_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.base_model import DGMFA
from dataloader import load_test_dataset

# =====================================================================
# Setup
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 120,
})

FOLD_COLORS = ['#2563eb', '#e11d48', '#8b5cf6', '#059669', '#f59e0b',
               '#ea580c', '#0ea5e9', '#a855f7', '#14b8a6', '#dc2626']
METRIC_COLORS = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db",
                 "#71b7ed", "#b8aeeb", "#f2a7da", "#C3D9B1", "#f4a261"]


# =====================================================================
# Metrics
# =====================================================================
def _specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_metrics(y_true, y_prob, thr):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spc = _specificity(y_true, y_pred)
    return {
        "threshold":   float(thr),
        "acc":         accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      rec,
        "specificity": spc,
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "auc":         roc_auc_score(y_true, y_prob),
        "auprc":       average_precision_score(y_true, y_prob),
        "mcc":         matthews_corrcoef(y_true, y_pred),
        "ba":          balanced_accuracy_score(y_true, y_pred),
        "gmean":       float(np.sqrt(max(rec, 0.0) * max(spc, 0.0))),
    }


def sweep_threshold(y_true, y_prob, grid=1001):
    thr_grid = np.linspace(0.0, 1.0, grid)
    mccs, f1s, bas = [], [], []
    for t in thr_grid:
        p = (y_prob >= t).astype(int)
        mccs.append(matthews_corrcoef(y_true, p))
        f1s.append(f1_score(y_true, p, zero_division=0))
        bas.append(balanced_accuracy_score(y_true, p))
    mccs, f1s, bas = map(np.array, (mccs, f1s, bas))
    return thr_grid, mccs, f1s, bas, float(thr_grid[int(np.argmax(mccs))])


# =====================================================================
# Bootstrap CI
# =====================================================================
def bootstrap_ci(y_true, y_prob, thr, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    n = len(y_true)
    collect = {k: [] for k in ["acc", "precision", "recall", "specificity",
                               "f1", "auc", "auprc", "mcc", "ba", "gmean"]}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            m = compute_metrics(yt, yp, thr)
            for k in collect:
                collect[k].append(m[k])
        except Exception:
            continue
    lo_q = (1 - ci) / 2; hi_q = 1 - lo_q
    out = {}
    for k, vals in collect.items():
        if len(vals) == 0:
            out[k] = (np.nan, np.nan, np.nan)
        else:
            v = np.asarray(vals)
            out[k] = (float(v.mean()),
                      float(np.quantile(v, lo_q)),
                      float(np.quantile(v, hi_q)))
    return out


# =====================================================================
# Checkpoint discovery  (V8 layout)
# =====================================================================
def resolve_model_dir(save_dir, dataset, mod_str=None, hp_str=None):
    ds_root = os.path.join(save_dir, dataset)
    if not os.path.isdir(ds_root):
        raise FileNotFoundError(f"{ds_root} does not exist")

    def _list_subdirs(path):
        return sorted(d for d in os.listdir(path)
                      if os.path.isdir(os.path.join(path, d)))

    def _has_folds(path):
        return os.path.isdir(path) and any(
            fn.startswith("fold") and fn.endswith(".pt")
            for fn in os.listdir(path)
        )

    if mod_str is not None and hp_str is not None:
        p = os.path.join(ds_root, mod_str, hp_str)
        if _has_folds(p):
            return p
        raise FileNotFoundError(f"{p} has no fold checkpoints.")

    if mod_str is not None and hp_str is None:
        mod_root = os.path.join(ds_root, mod_str)
        if not os.path.isdir(mod_root):
            raise FileNotFoundError(f"{mod_root} does not exist")
        hps = [d for d in _list_subdirs(mod_root) if _has_folds(os.path.join(mod_root, d))]
        if len(hps) == 1:
            return os.path.join(mod_root, hps[0])
        if len(hps) == 0:
            if _has_folds(mod_root):
                return mod_root
            raise FileNotFoundError(f"No fold checkpoints under {mod_root}")
        raise ValueError(
            f"Multiple hp subdirs under {mod_root}: {hps}. "
            f"Pass --hp_str <one>."
        )

    mods = _list_subdirs(ds_root)
    candidates = []
    for m in mods:
        m_path = os.path.join(ds_root, m)
        hps = [d for d in _list_subdirs(m_path) if _has_folds(os.path.join(m_path, d))]
        for h in hps:
            candidates.append((m, h))
        if _has_folds(m_path) and len(hps) == 0:
            candidates.append((m, None))

    if _has_folds(ds_root):
        candidates.append((None, None))

    if len(candidates) == 1:
        m, h = candidates[0]
        if m is None and h is None:
            return ds_root
        if h is None:
            return os.path.join(ds_root, m)
        return os.path.join(ds_root, m, h)

    if len(candidates) == 0:
        raise FileNotFoundError(f"No fold checkpoints under {ds_root}")

    msg = "Multiple configurations found, please pass --mod_str (and optionally --hp_str):\n"
    for m, h in candidates:
        msg += f"  mod_str={m!r}  hp_str={h!r}\n"
    raise ValueError(msg)


def find_checkpoint(model_dir, fold_idx_1based):
    ckpt_candidates = [
        f"fold{fold_idx_1based}_best_model.pt",
        f"best_fold{fold_idx_1based}.pt",
        f"model_fold{fold_idx_1based}.pt",
    ]
    ckpt_path = None
    for nm in ckpt_candidates:
        p = os.path.join(model_dir, nm)
        if os.path.exists(p):
            ckpt_path = p; break
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint for fold {fold_idx_1based} in {model_dir}. "
            f"Tried: {ckpt_candidates}"
        )

    thr_candidates = [f"fold{fold_idx_1based}_best_thr.npy"]
    thr_path = None
    for nm in thr_candidates:
        p = os.path.join(model_dir, nm)
        if os.path.exists(p):
            thr_path = p; break
    return ckpt_path, thr_path


# =====================================================================
# Inference for one fold
# =====================================================================
@torch.inference_mode()
def run_one_fold(fold_1based, test_loader, model_kwargs, model_dir):
    model = DGMFA(**model_kwargs).to(DEVICE)

    ckpt_path, thr_path = find_checkpoint(model_dir, fold_1based)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print(f"[Fold {fold_1based}] Warning: "
              f"{len(missing)} missing, {len(unexpected)} unexpected keys.")

    if thr_path is None:
        print(f"[Fold {fold_1based}] No saved threshold, using 0.5.")
        thr = 0.5
    else:
        thr = float(np.load(thr_path))

    model.eval()
    ys, ps = [], []
    for batch in test_loader:
        x1, x2, esm_tok, esm_pool, y, mask = batch
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
        esm_tok, esm_pool = esm_tok.to(DEVICE), esm_pool.to(DEVICE)
        mask = mask.to(DEVICE).bool()
        logits = model(x1, x2, esm_tok, esm_pool, mask)
        prob = torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
        ys.extend(y.numpy()); ps.extend(prob)
    return np.asarray(ys).astype(int), np.asarray(ps).astype(float), thr, ckpt_path


# =====================================================================
# Plots
# =====================================================================
def plot_cv_roc(fold_preds, save_path, title):
    plt.figure(figsize=(6, 5.5))
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 200)
    for i, (yt, yp) in enumerate(fold_preds):
        fpr, tpr, _ = roc_curve(yt, yp)
        a = roc_auc_score(yt, yp); aucs.append(a)
        plt.plot(fpr, tpr, lw=1.6, alpha=0.55,
                 color=FOLD_COLORS[i % len(FOLD_COLORS)],
                 label=f"Fold {i + 1} (AUC={a:.3f})")
        tpr_i = np.interp(mean_fpr, fpr, tpr); tpr_i[0] = 0.0
        tprs.append(tpr_i)
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color="#111827", lw=2.6,
             label=f"Mean (AUC={np.mean(aucs):.3f} +/- {np.std(aucs):.3f})")
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0),
                     np.minimum(mean_tpr + std_tpr, 1),
                     color="#111827", alpha=0.15, label="+/- 1 std")
    plt.plot([0, 1], [0, 1], ls='--', color="#9ca3af")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right", fontsize=9); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(save_path, dpi=220); plt.close()


def plot_cv_pr(fold_preds, save_path, title):
    plt.figure(figsize=(6, 5.5))
    aps = []
    for i, (yt, yp) in enumerate(fold_preds):
        p, r, _ = precision_recall_curve(yt, yp)
        a = average_precision_score(yt, yp); aps.append(a)
        plt.plot(r, p, lw=1.6, alpha=0.6,
                 color=FOLD_COLORS[i % len(FOLD_COLORS)],
                 label=f"Fold {i + 1} (AP={a:.3f})")
    plt.title(title + f"\nmean AP = {np.mean(aps):.3f} +/- {np.std(aps):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(loc="lower left", fontsize=9); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(save_path, dpi=220); plt.close()


def plot_confusion(y_true, y_pred, save_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-ACP", "ACP"])
    ax.set_yticklabels(["Non-ACP", "ACP"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > vmax / 2 else "black", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(save_path, dpi=220); plt.close(fig)


def plot_prob_hist(y_true, y_prob, thr, save_path, title):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    plt.figure(figsize=(6.2, 4))
    bins = np.linspace(0, 1, 31)
    plt.hist(y_prob[y_true == 0], bins=bins, alpha=0.65, label="Non-ACP", color="#6baed6")
    plt.hist(y_prob[y_true == 1], bins=bins, alpha=0.65, label="ACP", color="#fb6a4a")
    plt.axvline(thr, ls="--", color="black", lw=1.3, label=f"thr = {thr:.2f}")
    plt.xlabel("Predicted probability (ACP)"); plt.ylabel("Count")
    plt.title(title); plt.legend(); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(save_path, dpi=220); plt.close()


def plot_calibration(y_true, y_prob, save_path, title, n_bins=10):
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    accs, confs, sizes = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            accs.append(y_true[m].mean()); confs.append(y_prob[m].mean()); sizes.append(int(m.sum()))
        else:
            accs.append(0); confs.append((bins[b] + bins[b + 1]) / 2); sizes.append(0)
    total = sum(sizes)
    ece = sum(sizes[b] / max(total, 1) * abs(accs[b] - confs[b]) for b in range(n_bins))
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], '--', color="#9ca3af", label="Perfect calibration")
    ax.plot(confs, accs, "o-", color="#1f77b4", lw=2.2, markersize=7, label="Model")
    ax2 = ax.twinx()
    ax2.bar(bins[:-1] + 0.5 / n_bins, sizes, width=0.9 / n_bins,
            alpha=0.15, color="#9ca3af")
    ax2.set_ylabel("Bin count", color="#6b7280")
    ax.set_xlabel("Confidence (mean predicted prob.)"); ax.set_ylabel("Accuracy (empirical)")
    ax.set_title(f"{title}\nECE = {ece:.3f}"); ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(save_path, dpi=220); plt.close(fig)


def plot_threshold_sweep(y_true, y_prob, best_thr_val, save_path, title):
    thr_grid, mccs, f1s, bas, best_test = sweep_threshold(y_true, y_prob)
    plt.figure(figsize=(7, 4.5))
    plt.plot(thr_grid, mccs, lw=2.0, color="#d62728", label="MCC")
    plt.plot(thr_grid, f1s,  lw=2.0, color="#2ca02c", label="F1")
    plt.plot(thr_grid, bas,  lw=2.0, color="#1f77b4", label="BA")
    plt.axvline(best_thr_val, ls="--", color="#8b5cf6", lw=1.5,
                label=f"val-best thr = {best_thr_val:.2f}")
    plt.axvline(best_test, ls=":", color="#111827", lw=1.5,
                label=f"test-best MCC thr = {best_test:.2f}")
    plt.axvline(0.5, ls="--", color="#9ca3af", lw=1.0, label="thr = 0.5")
    plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title(title)
    plt.legend(loc="lower center", fontsize=9, ncol=3); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(save_path, dpi=220); plt.close()


def plot_per_fold_heatmap(df, save_path, title):
    metrics = [c for c in ["acc", "precision", "recall", "specificity",
                           "f1", "auc", "auprc", "mcc", "ba", "gmean"]
               if c in df.columns]
    mat = df[metrics].values
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(metrics) + 2), 0.5 * len(df) + 2))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_yticks(range(len(df)));      ax.set_yticklabels([f"Fold {i + 1}" for i in range(len(df))])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > 0.6 else "black", fontsize=9)
    ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(save_path, dpi=220); plt.close(fig)


def plot_metric_bars(df, save_path, title):
    metrics = [c for c in ["acc", "precision", "recall", "specificity",
                           "f1", "auc", "auprc", "mcc", "ba", "gmean"]
               if c in df.columns]
    means = [df[m].mean() for m in metrics]
    stds  = [df[m].std()  for m in metrics]
    fig, ax = plt.subplots(figsize=(max(10, 1.05 * len(metrics)), 6.4))
    bars = ax.bar(metrics, means, yerr=stds, capsize=5,
                  color=METRIC_COLORS[:len(metrics)], alpha=0.92,
                  edgecolor="#1E1E1E", linewidth=1.1)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{mean:.3f}+/-{std:.3f}",
                ha="center", va="bottom", fontsize=9.5, color="#222")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12); ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.grid(axis="y", ls="--", alpha=0.3, zorder=0)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for bar in bars: bar.set_zorder(3)
    fig.tight_layout(); fig.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_three_regime_bars(rows, save_path, title):
    df = pd.DataFrame(rows)
    metrics = [c for c in ["acc", "f1", "auc", "auprc", "mcc", "ba"]
               if (c + "_mean") in df.columns]
    regimes = df["regime"].unique().tolist()
    x = np.arange(len(metrics)); w = 0.8 / max(len(regimes), 1)
    colors = {"val-best": "#4c72b0", "thr=0.5": "#dd8452", "test-best*": "#55a868"}
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, reg in enumerate(regimes):
        sub = df[df["regime"] == reg]
        means = [float(sub[m + "_mean"].values[0]) for m in metrics]
        stds  = [float(sub[m + "_std"].values[0])  for m in metrics]
        ax.bar(x + (i - (len(regimes) - 1) / 2) * w, means, w,
               yerr=stds, capsize=3,
               label=reg, color=colors.get(reg, "#999"),
               edgecolor="#222", linewidth=0.8, alpha=0.92)
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05); ax.set_title(title)
    ax.legend(ncol=3, loc="lower center")
    ax.grid(axis="y", ls="--", alpha=0.3)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(save_path, dpi=240); plt.close(fig)


def plot_ensemble_comparison(perf_rows, save_path, title):
    df = pd.DataFrame(perf_rows)
    metrics = [c for c in ["acc", "f1", "auc", "auprc", "mcc", "ba"] if c in df.columns]
    x = np.arange(len(metrics)); w = 0.8 / len(df)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, row in df.iterrows():
        vals = [row[m] for m in metrics]
        ax.bar(x + (i - (len(df) - 1) / 2) * w, vals, w,
               label=row["name"], color=METRIC_COLORS[i % len(METRIC_COLORS)],
               edgecolor="#222", linewidth=0.8, alpha=0.92)
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05); ax.set_title(title); ax.set_ylabel("Score")
    ax.legend(ncol=len(df), loc="lower center")
    ax.grid(axis="y", ls="--", alpha=0.3)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(save_path, dpi=240); plt.close(fig)


def plot_bootstrap_ci(ci_dict, save_path, title):
    _skip = {"threshold"}
    metrics = [k for k in ci_dict.keys() if k not in _skip]
    means = [ci_dict[k][0] for k in metrics]
    los   = [ci_dict[k][1] for k in metrics]
    his   = [ci_dict[k][2] for k in metrics]
    err_lo = [max(m - l, 0.0) for m, l in zip(means, los)]
    err_hi = [max(h - m, 0.0) for m, h in zip(his, means)]
    fig, ax = plt.subplots(figsize=(max(9, 1.0 * len(metrics)), 5))
    ax.bar(metrics, means, yerr=[err_lo, err_hi], capsize=5,
           color=METRIC_COLORS[:len(metrics)], edgecolor="#222", alpha=0.92)
    for i, (m, lo, hi) in enumerate(zip(means, los, his)):
        ax.text(i, m + 0.01, f"{m:.3f}\n[{lo:.3f}, {hi:.3f}]",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, 1.15); ax.set_title(title); ax.set_ylabel("Score (95% CI)")
    ax.grid(axis="y", ls="--", alpha=0.3)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(save_path, dpi=240); plt.close(fig)


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser("ACP Evaluation V8.1 (with modality ablation)")

    # data / checkpoints
    parser.add_argument("--dataset", type=str, default="ACPMain",
                        choices=["ACPMain", "ACPAlternate", "ACP740", "DeepGram"])
    parser.add_argument("--save_dir",   type=str, default="checkpoints/V8_1_modality_ablation")
    parser.add_argument("--result_dir", type=str, default="results/V8_1_modality_ablation_eval")

    parser.add_argument("--mod_str", type=str, default=None,
                        help="e.g. BASE, noGRU, noTR, noFiLM, noBPF, noZS, noESM, "
                             "or combinations. If omitted and only one exists, auto-picked.")
    parser.add_argument("--hp_str",  type=str, default=None,
                        help="e.g. w96_do0.3_lr0.0001_sm0.1_gb1.0_gt1.5.")

    parser.add_argument("--folds",      type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len",    type=int, default=50)

    # model kwargs (MUST match training-time construction)
    parser.add_argument("--width",   type=int,   default=96)
    parser.add_argument("--dropout", type=float, default=0.3)

    # architecture-level ablations (must match training)
    parser.add_argument("--disable_bigru",       action="store_true")
    parser.add_argument("--disable_transformer", action="store_true")
    parser.add_argument("--disable_film",        action="store_true")
    parser.add_argument("--disable_gate",        action="store_true")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used at training time; used to locate the checkpoint directory.")
    

    # ===== NEW: modality-level ablations (must match training) =====
    parser.add_argument("--disable_bpf",    action="store_true",
                        help="Evaluate with BPF (x1) zeroed. MUST match training.")
    parser.add_argument("--disable_zscale", action="store_true",
                        help="Evaluate with Z-scale (x2) zeroed. MUST match training.")
    parser.add_argument("--disable_esm",    action="store_true",
                        help="Evaluate with ESM-2 zeroed. MUST match training.")
    # ================================================================

    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    # ---- safety check ----
    if args.disable_bpf and args.disable_zscale and args.disable_esm:
        raise ValueError("At least one input modality must remain enabled.")

    # Resolve checkpoint directory.
    # model_dir = resolve_model_dir(
    #     args.save_dir, args.dataset,
    #     mod_str=args.mod_str, hp_str=args.hp_str,
    # )

    model_dir = resolve_model_dir(
        os.path.join(args.save_dir, args.dataset, args.mod_str or "BASE",
                     f"seed{args.seed}"),
        dataset="",
        mod_str=None, hp_str=args.hp_str,
    )
    print(f"[INFO] model_dir = {model_dir}")

    # Result directory mirrors the model dir structure under result_dir.
    rel = os.path.relpath(model_dir, args.save_dir)
    result_dir = os.path.join(args.result_dir, rel)
    os.makedirs(result_dir, exist_ok=True)
    print(f"[INFO] result_dir = {result_dir}")
    print(f"[INFO] modality flags: bpf={not args.disable_bpf}, "
          f"zscale={not args.disable_zscale}, esm={not args.disable_esm}")

    # Data.
    test_path = f"data/{args.dataset}/test.csv"
    test_set = load_test_dataset(test_path, max_len=args.max_len)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Model kwargs.
    model_kwargs = dict(
        x1_channels=20, x2_channels=5, esm_token_dim=1280,
        width=args.width,
        use_bigru=not args.disable_bigru,
        use_transformer=not args.disable_transformer,
        use_film=not args.disable_film,
        use_gate=not args.disable_gate,
        # ===== NEW: modality flags =====
        use_bpf=not args.disable_bpf,
        use_zscale=not args.disable_zscale,
        use_esm=not args.disable_esm,
        # ===============================
        film_cond_dim=1280,
        max_len=args.max_len,
        dropout=args.dropout,
    )

    # Run every fold.
    per_fold_metrics_valbest = []
    per_fold_metrics_thr05 = []
    per_fold_metrics_testbest = []
    fold_preds = []
    fold_thrs = []
    y_true_ref = None

    for i in range(1, args.folds + 1):
        yt, yp, thr_val, ckpt_path = run_one_fold(i, test_loader, model_kwargs, model_dir)
        print(f"[Fold {i}] ckpt = {os.path.basename(ckpt_path)} | val-best thr = {thr_val:.3f}")
        if y_true_ref is None:
            y_true_ref = yt
        else:
            assert np.array_equal(y_true_ref, yt), "Test labels differ across folds!"
        fold_preds.append((yt, yp))
        fold_thrs.append(thr_val)

        per_fold_metrics_valbest.append(compute_metrics(yt, yp, thr_val))
        per_fold_metrics_thr05.append(compute_metrics(yt, yp, 0.5))
        _, _, _, _, thr_test_best = sweep_threshold(yt, yp)
        per_fold_metrics_testbest.append(compute_metrics(yt, yp, thr_test_best))

    # Per-fold CSVs.
    df_valbest  = pd.DataFrame(per_fold_metrics_valbest)
    df_thr05    = pd.DataFrame(per_fold_metrics_thr05)
    df_testbest = pd.DataFrame(per_fold_metrics_testbest)
    df_valbest.to_csv(os.path.join(result_dir, "per_fold_valbest.csv"),  index_label="fold")
    df_thr05.to_csv(os.path.join(result_dir,   "per_fold_thr05.csv"),    index_label="fold")
    df_testbest.to_csv(os.path.join(result_dir, "per_fold_testbest.csv"), index_label="fold")

    # Summaries.
    def _summ(df):
        return pd.DataFrame({k: {"Mean": df[k].mean(), "StdDev": df[k].std()}
                             for k in df.columns}).T
    sdf_val = _summ(df_valbest)
    sdf_05  = _summ(df_thr05)
    sdf_tb  = _summ(df_testbest)
    sdf_val.to_csv(os.path.join(result_dir, "summary_valbest.csv"))
    sdf_05.to_csv(os.path.join(result_dir,  "summary_thr05.csv"))
    sdf_tb.to_csv(os.path.join(result_dir,  "summary_testbest.csv"))

    print(f"\n===== {args.dataset} | val-best threshold (MAIN RESULT) =====")
    print(sdf_val.to_string(float_format="{:.4f}".format))

    # ROC / PR overlays.
    plot_cv_roc(fold_preds, os.path.join(result_dir, "cv_roc_overlay.png"),
                title=f"ROC on test set - {args.dataset}")
    plot_cv_pr(fold_preds, os.path.join(result_dir, "cv_pr_overlay.png"),
               title=f"PR on test set - {args.dataset}")

    # Per-fold prob histograms.
    for i, (yt, yp) in enumerate(fold_preds):
        plot_prob_hist(yt, yp, fold_thrs[i],
                       os.path.join(result_dir, f"fold{i+1}_prob_hist.png"),
                       title=f"Fold {i+1} - Predicted probability")

    # Ensembles.
    P = np.stack([yp for (_, yp) in fold_preds], axis=0)
    mean_prob = P.mean(axis=0)
    votes = np.stack([(P[i] >= fold_thrs[i]).astype(int) for i in range(args.folds)], axis=0)
    majority = (votes.sum(axis=0) >= (args.folds / 2.0)).astype(int)

    avg_thr = float(np.mean(fold_thrs))
    m_meanprob = compute_metrics(y_true_ref, mean_prob, avg_thr)
    m_majority = compute_metrics(y_true_ref, majority.astype(float), 0.5)

    with open(os.path.join(result_dir, "ensemble_metrics.json"), "w") as f:
        json.dump({"mean_prob@avg_thr": m_meanprob,
                   "avg_thr_used": avg_thr,
                   "majority_vote": m_majority}, f, indent=2)

    plot_ensemble_comparison(
        [{"name": "Mean probability", **m_meanprob},
         {"name": "Majority vote",    **m_majority}],
        os.path.join(result_dir, "ensemble_compare.png"),
        title=f"Ensemble comparison - {args.dataset}"
    )

    # Per-fold metric heatmap.
    plot_per_fold_heatmap(df_valbest,
                          os.path.join(result_dir, "per_fold_heatmap_valbest.png"),
                          title=f"Per-fold metrics (val-best thr) - {args.dataset}")

    # Three-regime bar plot.
    rows = []
    for name, df in [("val-best", df_valbest),
                     ("thr=0.5",  df_thr05),
                     ("test-best*", df_testbest)]:
        rec = {"regime": name}
        for c in df.columns:
            rec[c + "_mean"] = df[c].mean()
            rec[c + "_std"]  = df[c].std()
        rows.append(rec)
    plot_three_regime_bars(rows,
                           os.path.join(result_dir, "threshold_regimes_compare.png"),
                           title=f"Threshold regimes - {args.dataset}\n"
                                 "(*test-best is an upper bound, NOT for main reporting)")

    # Main bar plot.
    plot_metric_bars(df_valbest,
                     os.path.join(result_dir, "metrics_barplot_valbest.png"),
                     title=f"{args.dataset} | 5-Fold Test Performance (val-best thr)")

    # Mean-prob based plots.
    plot_confusion(y_true_ref, (mean_prob >= avg_thr).astype(int),
                   os.path.join(result_dir, "confusion_valbest_meanprob.png"),
                   title=f"Confusion (mean-prob, thr={avg_thr:.2f}) - {args.dataset}")
    plot_confusion(y_true_ref, (mean_prob >= 0.5).astype(int),
                   os.path.join(result_dir, "confusion_thr05_meanprob.png"),
                   title=f"Confusion (mean-prob, thr=0.5) - {args.dataset}")
    plot_prob_hist(y_true_ref, mean_prob, avg_thr,
                   os.path.join(result_dir, "prob_hist_meanprob.png"),
                   title=f"Mean probability - {args.dataset}")
    plot_calibration(y_true_ref, mean_prob,
                     os.path.join(result_dir, "calibration_meanprob.png"),
                     title=f"Reliability diagram (mean-prob) - {args.dataset}")
    plot_threshold_sweep(y_true_ref, mean_prob, avg_thr,
                         os.path.join(result_dir, "threshold_sweep_meanprob.png"),
                         title=f"Threshold sweep (mean-prob) - {args.dataset}")

    # Bootstrap 95% CI.
    ci = bootstrap_ci(y_true_ref, mean_prob, avg_thr, n_boot=args.n_boot)
    ci_df = pd.DataFrame(ci, index=["mean", "lo", "hi"]).T
    ci_df.to_csv(os.path.join(result_dir, "bootstrap_ci_meanprob.csv"))
    plot_bootstrap_ci(ci, os.path.join(result_dir, "bootstrap_ci_meanprob.png"),
                      title=f"Bootstrap 95% CI (mean-prob, n={args.n_boot}) - {args.dataset}")

    # Save eval config.
    cfg = dict(vars(args))
    cfg.update({
        "model_dir": model_dir,
        "avg_val_best_thr": avg_thr,
        "fold_thrs": fold_thrs,
        "n_test_samples": int(len(y_true_ref)),
        "n_pos": int(y_true_ref.sum()),
        "n_neg": int((1 - y_true_ref).sum()),
    })
    with open(os.path.join(result_dir, "eval_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    print(f"\n[Done] Results written to: {result_dir}")


if __name__ == "__main__":
    main()
