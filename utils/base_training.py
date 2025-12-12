# utils/base_training.py
# -*- coding: utf-8 -*-
import os, numpy as np, torch, math
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------- Helpers -------------------------
def _compute_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def _compute_metrics(y_true, y_prob, thr):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)

    rec = recall_score(y_true, y_pred)
    spc = _compute_specificity(y_true, y_pred)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': rec,
        'specificity': spc,
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'ba': (rec + spc) / 2.0,
        'gmean': float(np.sqrt(max(rec, 0.0) * max(spc, 0.0)))
    }

def _best_mcc_threshold(y_true, y_prob, grid_points=1001):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thr_grid = np.linspace(0.0, 1.0, grid_points)
    mccs = []
    for t in thr_grid:
        pred = (y_prob >= t).astype(int)
        mccs.append(matthews_corrcoef(y_true, pred))
    mccs = np.asarray(mccs)
    idx = int(np.argmax(mccs))
    return float(thr_grid[idx]), float(mccs[idx]), thr_grid, mccs

def _plot_roc(y_true, y_prob, save_path, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2.0, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1], '--', color="#9ca3af")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title)
    plt.legend(loc='lower right'); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()

def _plot_mcc_curve(thr_grid, mccs, best_thr, best_mcc, save_path, title="MCC vs Threshold"):
    plt.figure(figsize=(6,5))
    plt.plot(thr_grid, mccs, lw=2.0)
    plt.axvline(best_thr, color="#ef4444", lw=1.5, ls='--', label=f"best_thr={best_thr:.3f}")
    plt.axhline(best_mcc, color="#10b981", lw=1.5, ls='--', label=f"best_MCC={best_mcc:.3f}")
    plt.xlabel("Threshold"); plt.ylabel("MCC"); plt.title(title)
    plt.legend(); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


class SmoothBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.0, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        # targets: [B,1]
        with torch.no_grad():
            smoothed_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce(logits, smoothed_targets)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ------------------------- Early Stopping -------------------------
class EarlyStopping:

    def __init__(self, patience=15, mode='max', delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best = -float('inf') if mode == 'max' else float('inf')
        self.counter = 0

    def step(self, score):
        improved = (score > self.best + self.delta) if self.mode == 'max' else (score < self.best - self.delta)
        if improved:
            self.best = score
            self.counter = 0
            return False  # not stop
        else:
            self.counter += 1
            return self.counter >= self.patience


# ------------------------- Schedulers -------------------------
class WarmupCosine:

    def __init__(self, optimizer, base_lr, total_epochs, warmup_epochs=5):
        self.opt = optimizer
        self.base_lr = base_lr
        self.total = total_epochs
        self.warmup = warmup_epochs
        self.epoch = 0

        for pg in self.opt.param_groups:
            pg['lr'] = 1e-8

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup:
            scale = self.epoch / max(self.warmup, 1)
            lr = self.base_lr * scale
        else:
            # t in [0, 1]
            t = (self.epoch - self.warmup) / max(self.total - self.warmup, 1)
            lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * t))
        for pg in self.opt.param_groups:
            pg['lr'] = lr

    def get_last_lr(self):
        return [pg['lr'] for pg in self.opt.param_groups]


# ------------------------- Train One Fold -------------------------
def train_one_fold(model, train_loader, val_loader, fold, save_dir, device,
                      base_lr=2e-4, weight_decay=1e-4,
                      total_epochs=100, warmup_epochs=5,
                      label_smoothing=0.05, grad_clip=5.0,
                      amp=True):
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = WarmupCosine(optimizer, base_lr=base_lr, total_epochs=total_epochs, warmup_epochs=warmup_epochs)

    try:
        criterion = nn.BCEWithLogitsLoss(label_smoothing=label_smoothing)
    except TypeError:
        criterion = SmoothBCEWithLogitsLoss(smoothing=label_smoothing)

    es = EarlyStopping(patience=15, mode='max', delta=1e-4)
    #scaler = GradScaler(enabled=amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_mcc = -1e9
    best_thr = 0.5
    best_path = os.path.join(save_dir, f'fold{fold}_best_model.pt')
    thr_path  = os.path.join(save_dir, f'fold{fold}_best_thr.npy')

    for epoch in range(1, total_epochs + 1):
        # ---------------- train ----------------
        model.train()
        for batch in train_loader:
            # dataloader_v3: (x1, x2, esm_tok, esm_pool, y, mask)
            x1, x2, esm_tok, esm_pool, y, mask = batch
            x1, x2 = x1.to(device), x2.to(device)
            esm_tok, esm_pool = esm_tok.to(device), esm_pool.to(device)
            y = y.to(device).float().unsqueeze(1)
            mask = mask.to(device).bool()

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x1, x2, esm_tok, esm_pool, mask)  # [B,1]
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # ---------------- validate ----------------
        model.eval()
        ys, ps = [], []
        with torch.inference_mode():
            for batch in val_loader:
                x1, x2, esm_tok, esm_pool, y, mask = batch
                x1, x2 = x1.to(device), x2.to(device)
                esm_tok, esm_pool = esm_tok.to(device), esm_pool.to(device)
                mask = mask.to(device).bool()
                logits = model(x1, x2, esm_tok, esm_pool, mask)  # [B,1]
                prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                ys.extend(y.numpy())
                ps.extend(prob)


        epoch_thr, epoch_mcc, thr_grid, mccs = _best_mcc_threshold(ys, ps)

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
            print(f"[Fold {fold}] Epoch {epoch} | best_MCC={best_mcc:.4f} | thr*={best_thr:.3f} | lr={scheduler.get_last_lr()[0]:.2e}")

        # early stopping
        if es.step(epoch_mcc):
            print(f"[Fold {fold}] Early stop at epoch {epoch} | best_MCC={best_mcc:.4f}")
            break

    # }e best CÍ
    model.load_state_dict(torch.load(best_path, map_location=device))


    return ys, ps, _compute_metrics(ys, ps, best_thr)


# ------------------------- Saving per-fold & summary -------------------------
def save_fold_results(metrics, save_dir, fold, y_true, y_prob):
    os.makedirs(save_dir, exist_ok=True)

    df = {k: [v] for k, v in metrics.items()}
    pd.DataFrame(df).to_csv(os.path.join(save_dir, f'fold{fold}_metrics.csv'), index=False)
    # þ
    _plot_roc(y_true, y_prob, os.path.join(save_dir, f'fold{fold}_roc_final.png'),
              title=f"ROC (final probs) - Fold {fold}")
    thr, mcc, thr_grid, mccs = _best_mcc_threshold(y_true, y_prob)
    _plot_mcc_curve(thr_grid, mccs, thr, mcc, os.path.join(save_dir, f'fold{fold}_mcc_thr_final.png'),
                    title=f"MCC vs Threshold (final probs) - Fold {fold}")

def save_final_summary(fold_metrics, save_dir, also_fixed_thr=False, fixed_thr=0.5):

    os.makedirs(save_dir, exist_ok=True)
    # G<±¹î
    keys = sorted(fold_metrics[0].keys())
    summary = {k: {"Mean": np.mean([m[k] for m in fold_metrics]),
                   "StdDev": np.std([m[k] for m in fold_metrics])}
               for k in keys}
    df = pd.DataFrame(summary).T
    df.to_csv(os.path.join(save_dir, 'summary_metrics.csv'))
    print("===== Final 5-Fold Summary (best thr per fold) =====")
    print(df.round(4))

    if also_fixed_thr:
        print(f"\n(Info) To report fixed threshold={fixed_thr}, "
              "please pass another list of fold_metrics computed with that fixed threshold.")
