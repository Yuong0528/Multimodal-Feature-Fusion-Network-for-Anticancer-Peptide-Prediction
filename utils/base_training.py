# utils/base_training_v8.py
# -*- coding: utf-8 -*-
"""
Training utilities for DGMFA
"""
import os, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 120,
})


# -----------------------------------------------------------------
# Loss
# -----------------------------------------------------------------
class SmoothBCEWithLogitsLoss(nn.Module):
    """Fallback if torch doesn't accept label_smoothing on BCE."""
    def __init__(self, smoothing=0.0, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        with torch.no_grad():
            smoothed = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce(logits, smoothed)
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum':  return loss.sum()
        return loss


# -----------------------------------------------------------------
# Param groups: no weight decay on bias / norm layers
# -----------------------------------------------------------------
def build_param_groups(model, weight_decay):
    decay, no_decay = [], []
    seen = set()
    norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                  nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)
    for mod in model.modules():
        for name, p in mod.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            if isinstance(mod, norm_types) or name == "bias" or p.ndim <= 1:
                no_decay.append(p)
            else:
                decay.append(p)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# -----------------------------------------------------------------
# Scheduler: warmup + cosine, starting at base_lr * 0.01
# -----------------------------------------------------------------
class WarmupCosine:
    def __init__(self, optimizer, base_lr, total_epochs, warmup_epochs=5,
                 min_lr_ratio=0.01):
        self.opt = optimizer
        self.base_lr = base_lr
        self.total = total_epochs
        self.warmup = warmup_epochs
        self.min_lr = base_lr * min_lr_ratio
        self.epoch = 0
        for pg in self.opt.param_groups:
            pg['lr'] = self.min_lr     # start from min_lr, not 1e-8

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup:
            scale = self.epoch / max(self.warmup, 1)
            lr = self.min_lr + (self.base_lr - self.min_lr) * scale
        else:
            t = (self.epoch - self.warmup) / max(self.total - self.warmup, 1)
            t = min(max(t, 0.0), 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))
        for pg in self.opt.param_groups:
            pg['lr'] = lr

    def get_last_lr(self):
        return [pg['lr'] for pg in self.opt.param_groups]


# -----------------------------------------------------------------
# Early stopping
# -----------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=20, mode='max', delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best = -float('inf') if mode == 'max' else float('inf')
        self.counter = 0

    def step(self, score):
        improved = (score > self.best + self.delta) if self.mode == 'max' \
                                                    else (score < self.best - self.delta)
        if improved:
            self.best = score; self.counter = 0; return False
        self.counter += 1
        return self.counter >= self.patience


# -----------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------
def _specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def _compute_metrics(y_true, y_prob, thr):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spc = _specificity(y_true, y_pred)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': rec,
        'specificity': spc,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'ba': (rec + spc) / 2.0,
        'gmean': float(np.sqrt(max(rec, 0.0) * max(spc, 0.0))),
    }

def _best_mcc_threshold(y_true, y_prob, grid_points=1001):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thr_grid = np.linspace(0.0, 1.0, grid_points)
    mccs = np.array([matthews_corrcoef(y_true, (y_prob >= t).astype(int))
                     for t in thr_grid])
    idx = int(np.argmax(mccs))
    return float(thr_grid[idx]), float(mccs[idx]), thr_grid, mccs


# -----------------------------------------------------------------
# Plots
# -----------------------------------------------------------------
def _plot_roc(y_true, y_prob, save_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, lw=2, color="#1f77b4", label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], '--', color="#9ca3af")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title)
    plt.legend(loc="lower right"); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()

def _plot_mcc_curve(thr_grid, mccs, best_thr, best_mcc, save_path, title):
    plt.figure(figsize=(5.5, 5))
    plt.plot(thr_grid, mccs, lw=2, color="#2ca02c")
    plt.axvline(best_thr, color="#ef4444", lw=1.5, ls='--',
                label=f"thr* = {best_thr:.3f}")
    plt.axhline(best_mcc, color="#10b981", lw=1.5, ls='--',
                label=f"MCC* = {best_mcc:.3f}")
    plt.xlabel("Threshold"); plt.ylabel("MCC"); plt.title(title)
    plt.legend(); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()

def _plot_training_curves(history, save_path, fold):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color="#1f77b4", lw=1.8, label="Train loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"(a) Training loss - Fold {fold}"); ax.legend(); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(epochs, history["val_auc"], color="#2ca02c", lw=1.8, label="Val AUC")
    ax.plot(epochs, history["val_mcc"], color="#d62728", lw=1.8, label="Val MCC")
    ax.plot(epochs, history["val_f1"],  color="#9467bd", lw=1.8, label="Val F1")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title(f"(b) Validation metrics - Fold {fold}"); ax.legend(); ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(epochs, history["lr"], color="#ff7f0e", lw=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    ax.set_title(f"(c) LR schedule - Fold {fold}"); ax.grid(alpha=0.25)

    ax = axes[1, 1]
    means = [x for x in history["gate_alpha_mean"] if x is not None]
    stds  = [x for x in history["gate_alpha_std"]  if x is not None]
    if len(means) > 0:
        ep = np.arange(1, len(means) + 1)
        means = np.array(means); stds = np.array(stds)
        ax.plot(ep, means, color="#17becf", lw=1.8, label="mean alpha")
        ax.fill_between(ep, means - stds, means + stds, alpha=0.2,
                        color="#17becf", label="+/-1 std")
        ax.set_ylim(0, 1); ax.set_xlabel("Epoch"); ax.set_ylabel("Gate alpha")
        ax.set_title(f"(d) Gate alpha (BiGRU weight) - Fold {fold}")
        ax.legend(); ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "(gate not used)", ha="center", va="center"); ax.axis("off")

    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


# -----------------------------------------------------------------
# Train one fold
# -----------------------------------------------------------------
def train_one_fold(model, train_loader, val_loader, fold, save_dir, device,
                   base_lr=1e-4, weight_decay=1e-4,
                   total_epochs=60, warmup_epochs=5,
                   label_smoothing=0.1, grad_clip=5.0,
                   amp=True, early_patience=20):
    os.makedirs(save_dir, exist_ok=True)

    param_groups = build_param_groups(model, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr)
    scheduler = WarmupCosine(optimizer, base_lr=base_lr,
                             total_epochs=total_epochs, warmup_epochs=warmup_epochs)
    try:
        criterion = nn.BCEWithLogitsLoss(label_smoothing=label_smoothing)
    except TypeError:
        criterion = SmoothBCEWithLogitsLoss(smoothing=label_smoothing)

    es = EarlyStopping(patience=early_patience, mode='max', delta=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_mcc = -1e9
    best_thr = 0.5
    best_path = os.path.join(save_dir, f'fold{fold}_best_model.pt')
    thr_path  = os.path.join(save_dir, f'fold{fold}_best_thr.npy')

    history = {
        "train_loss": [], "val_auc": [], "val_mcc": [], "val_f1": [],
        "lr": [], "gate_alpha_mean": [], "gate_alpha_std": [],
    }

    for epoch in range(1, total_epochs + 1):
        # ---------- train ----------
        model.train()
        running_loss, running_n = 0.0, 0
        gate_vals = []

        for batch in train_loader:
            x1, x2, esm_tok, esm_pool, y, mask = batch
            x1, x2 = x1.to(device), x2.to(device)
            esm_tok, esm_pool = esm_tok.to(device), esm_pool.to(device)
            y = y.to(device).float().unsqueeze(1)
            mask = mask.to(device).bool()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x1, x2, esm_tok, esm_pool, mask)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bs = y.shape[0]
            running_loss += float(loss.detach()) * bs
            running_n += bs

            if getattr(model, "last_gate_alpha", None) is not None:
                gate_vals.append(model.last_gate_alpha.detach().float().cpu().numpy().reshape(-1))

        scheduler.step()
        epoch_loss = running_loss / max(running_n, 1)
        cur_lr = scheduler.get_last_lr()[0]

        # ---------- validate ----------
        model.eval()
        ys, ps = [], []
        with torch.inference_mode():
            for batch in val_loader:
                x1, x2, esm_tok, esm_pool, y, mask = batch
                x1, x2 = x1.to(device), x2.to(device)
                esm_tok, esm_pool = esm_tok.to(device), esm_pool.to(device)
                mask = mask.to(device).bool()
                logits = model(x1, x2, esm_tok, esm_pool, mask)
                prob = torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
                ys.extend(y.numpy()); ps.extend(prob)

        epoch_thr, epoch_mcc, thr_grid, mccs = _best_mcc_threshold(ys, ps)
        epoch_auc = roc_auc_score(ys, ps)
        epoch_f1 = f1_score(np.asarray(ys).astype(int),
                            (np.asarray(ps) >= epoch_thr).astype(int),
                            zero_division=0)

        history["train_loss"].append(epoch_loss)
        history["val_auc"].append(epoch_auc)
        history["val_mcc"].append(epoch_mcc)
        history["val_f1"].append(epoch_f1)
        history["lr"].append(cur_lr)
        if len(gate_vals) > 0:
            all_g = np.concatenate(gate_vals, axis=0)
            history["gate_alpha_mean"].append(float(all_g.mean()))
            history["gate_alpha_std"].append(float(all_g.std()))
        else:
            history["gate_alpha_mean"].append(None)
            history["gate_alpha_std"].append(None)

        if epoch_mcc > best_mcc:
            best_mcc = epoch_mcc
            best_thr = epoch_thr
            torch.save(model.state_dict(), best_path)
            np.save(thr_path, best_thr)
            _plot_roc(ys, ps, os.path.join(save_dir, f"fold{fold}_roc.png"),
                      title=f"ROC - Fold {fold}")
            _plot_mcc_curve(thr_grid, mccs, best_thr, best_mcc,
                            os.path.join(save_dir, f"fold{fold}_mcc_thr.png"),
                            title=f"MCC vs Threshold - Fold {fold}")

        print(f"[Fold {fold}] Ep {epoch:3d} | loss={epoch_loss:.4f} | "
              f"AUC={epoch_auc:.4f} MCC={epoch_mcc:.4f} F1={epoch_f1:.4f} | "
              f"thr*={epoch_thr:.2f} best_MCC={best_mcc:.4f} | lr={cur_lr:.2e}")

        if es.step(epoch_mcc):
            print(f"[Fold {fold}] Early stop at epoch {epoch} | best_MCC={best_mcc:.4f}")
            break

    # save history + curves
    pd.DataFrame(history).to_csv(os.path.join(save_dir, f"fold{fold}_history.csv"),
                                 index=False)
    _plot_training_curves(history, os.path.join(save_dir, f"fold{fold}_training_curves.png"),
                          fold)

    # load best weights back for final eval
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for batch in val_loader:
            x1, x2, esm_tok, esm_pool, y, mask = batch
            x1, x2 = x1.to(device), x2.to(device)
            esm_tok, esm_pool = esm_tok.to(device), esm_pool.to(device)
            mask = mask.to(device).bool()
            logits = model(x1, x2, esm_tok, esm_pool, mask)
            prob = torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
            ys.extend(y.numpy()); ps.extend(prob)

    return ys, ps, _compute_metrics(ys, ps, best_thr)




# -----------------------------------------------------------------
# Save fold results
# -----------------------------------------------------------------
def save_fold_results(metrics, save_dir, fold, y_true, y_prob):
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame({k: [v] for k, v in metrics.items()}).to_csv(
        os.path.join(save_dir, f'fold{fold}_metrics.csv'), index=False)
    _plot_roc(y_true, y_prob, os.path.join(save_dir, f'fold{fold}_roc_final.png'),
              title=f"ROC (final) - Fold {fold}")
    thr, mcc, thr_grid, mccs = _best_mcc_threshold(y_true, y_prob)
    _plot_mcc_curve(thr_grid, mccs, thr, mcc,
                    os.path.join(save_dir, f'fold{fold}_mcc_thr_final.png'),
                    title=f"MCC vs Threshold (final) - Fold {fold}")


# -----------------------------------------------------------------
# Save final summary
# -----------------------------------------------------------------
def save_final_summary(fold_metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    keys = sorted(fold_metrics[0].keys())
    summary = {k: {"Mean": float(np.mean([m[k] for m in fold_metrics])),
                   "StdDev": float(np.std([m[k] for m in fold_metrics]))}
               for k in keys}
    df = pd.DataFrame(summary).T
    df.to_csv(os.path.join(save_dir, 'summary_metrics.csv'))
    print("===== Final 5-Fold Summary =====")
    print(df.round(4))

    # compact bar plot
    metric_order = ["acc", "precision", "recall", "specificity", "f1",
                    "auc", "auprc", "mcc", "ba", "gmean"]
    keys_plot = [k for k in metric_order if k in df.index]
    means = [df.loc[k, "Mean"] for k in keys_plot]
    stds  = [df.loc[k, "StdDev"] for k in keys_plot]

    colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db",
              "#71b7ed", "#b8aeeb", "#f2a7da", "#C3D9B1", "#f4a261"]
    plt.figure(figsize=(10, 4.8))
    plt.bar(keys_plot, means, yerr=stds, capsize=4,
            color=colors[:len(keys_plot)], edgecolor="#222", linewidth=0.9)
    for i, (m, s) in enumerate(zip(means, stds)):
        plt.text(i, m + 0.01, f"{m:.3f}+/-{s:.3f}",
                 ha="center", va="bottom", fontsize=9)
    plt.ylim(0, 1.08); plt.ylabel("Score")
    plt.title("5-Fold CV summary (best thr per fold)")
    plt.xticks(rotation=15); plt.grid(axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_bars.png'), dpi=220)
    plt.close()
