# models/base_model_v8_1.py
# -*- coding: utf-8 -*-
"""
Base model 
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =================================================================
# Basic building blocks
# =================================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):  # [B,C,L]
        scale = self.fc(x).unsqueeze(-1)
        return x * scale.expand_as(x)


class DilatedConvBlock(nn.Module):
    """For x2 (physicochemical, 5 channels)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MultiKernelConvBlock(nn.Module):
    """For x1 (one-hot, 20 channels)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # x:[B,C,L]
        B, C, L = x.shape
        return x + self.pe[:L, :C].T.unsqueeze(0)


# =================================================================
# Sequence models
# =================================================================

class BiGRUBlockResLN(nn.Module):
    def __init__(self, channels, hidden_size):
        super().__init__()
        self.gru = nn.GRU(channels, hidden_size,
                          batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_size * 2, channels)
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):  # [B,C,L]
        x_l = x.transpose(1, 2)
        y, _ = self.gru(x_l)
        y = self.proj(y)
        y = self.ln(y + x_l)
        return y.transpose(1, 2)


class TransformerBlock(nn.Module):
    """
    One Transformer encoder layer with padding-aware self-attention.
    dropout is now propagated from the constructor (V5 hard-coded 0.1).
    """
    def __init__(self, d_model=64, nhead=4, dim_ff=128, dropout=0.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(d_model, nhead,
                                        batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):  # [B,C,L]
        x_l = x.transpose(1, 2)  # [B,L,C]
        # PyTorch's MHA uses True=IGNORE; our convention is True=valid,
        # so we invert.
        kpm = (~key_padding_mask) if key_padding_mask is not None else None
        attn, _ = self.sa(x_l, x_l, x_l, key_padding_mask=kpm, need_weights=False)
        x_l = self.ln1(x_l + self.do(attn))
        ff = self.ff(x_l)
        x_l = self.ln2(x_l + self.do(ff))
        out = x_l.transpose(1, 2)  # [B,C,L]
        # Zero pad rows so pad queries don't leak non-zero activations.
        if key_padding_mask is not None:
            out = out * key_padding_mask.unsqueeze(1).float()
        return out


# =================================================================
# Cross-Attention (mask-hardened) and FiLM
# =================================================================

class CrossAttention(nn.Module):
    """
    Mask-hardened cross-attention:
      - AMP-safe negative fill (-1e4 instead of -inf);
      - zeros out output rows where all keys are pad (prevents NaN
        propagation from uniform-over-pad softmax);
      - masks pad queries on the output.
    """
    def __init__(self, q_dim, kv_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        assert q_dim % heads == 0, "q_dim must be divisible by heads"
        self.heads = heads
        self.scale = (q_dim // heads) ** -0.5
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.o_proj = nn.Linear(q_dim, out_dim)
        self.do = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, q, k, v, key_padding_mask=None, query_padding_mask=None):
        B, Lq, C = q.shape
        Lk = k.shape[1]
        H = self.heads

        qh = self.q_proj(q).view(B, Lq, H, C // H).transpose(1, 2)
        kh = self.k_proj(k).view(B, Lk, H, C // H).transpose(1, 2)
        vh = self.v_proj(v).view(B, Lk, H, C // H).transpose(1, 2)

        attn = (qh @ kh.transpose(-2, -1)) * self.scale  # [B,H,Lq,Lk]
        if key_padding_mask is not None:
            invalid = (~key_padding_mask).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(invalid, -1e4)       # [M1] AMP-safe

        attn_w = F.softmax(attn, dim=-1)
        if key_padding_mask is not None:
            any_valid = key_padding_mask.any(dim=-1)     # [M1] all-pad-key guard
            attn_w = attn_w * any_valid.view(B, 1, 1, 1).float()

        out = (attn_w @ vh).transpose(1, 2).contiguous().view(B, Lq, C)
        out = self.o_proj(self.do(out))
        out = self.ln(out + q)

        qmask = query_padding_mask if query_padding_mask is not None else key_padding_mask
        if qmask is not None:
            out = out * qmask.unsqueeze(-1).float()      # [M1] query-side mask
        return out


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_channels):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_channels)
        self.beta = nn.Linear(cond_dim, feat_channels)

    def forward(self, x, cond):  # x:[B,C,L]
        g = self.gamma(cond).unsqueeze(-1)
        b = self.beta(cond).unsqueeze(-1)
        return x * (1.0 + g) + b


# =================================================================
# Dual-branch parallel gate (BiGRU // Transformer)
# =================================================================

class DualBranchGate(nn.Module):
    def __init__(self, channels, init_bias=2.0, tau=1.0):
        super().__init__()
        self.tau = tau
        # input: concat(a_g, b_g, a_g - b_g) = 3*channels
        self.mlp_1 = nn.Linear(channels * 3, channels)
        self.mlp_2 = nn.Linear(channels, 1)
        # Init last layer: zero weights, positive bias -> alpha ~ 0.88.
        nn.init.zeros_(self.mlp_2.weight)
        nn.init.constant_(self.mlp_2.bias, init_bias)

    def forward(self, a, b):  # a, b: [B, C, L]
        a_g = F.adaptive_avg_pool1d(a, 1).squeeze(-1)   # [B, C]
        b_g = F.adaptive_avg_pool1d(b, 1).squeeze(-1)   # [B, C]
        diff = a_g - b_g
        h = self.mlp_1(torch.cat([a_g, b_g, diff], dim=1))
        h = F.gelu(h)
        logit = self.mlp_2(h) / self.tau                 # [B, 1]
        alpha = torch.sigmoid(logit).view(-1, 1, 1)      # [B, 1, 1]
        fused = alpha * a + (1.0 - alpha) * b
        return fused, alpha.squeeze(-1).squeeze(-1)      # [B]


# =================================================================
# Main model
# =================================================================

class DGMFA(nn.Module):
    def __init__(
        self,
        x1_channels=20, x2_channels=5, esm_token_dim=1280,
        width=96,
        use_bigru=True, use_transformer=True,
        use_film=True, use_gate=True,
        film_cond_dim=1280,
        max_len=512, dropout=0.3,
        gate_init_bias=2.0, gate_tau=1.0,   # [V8.1 new]
    ):
        super().__init__()
        self.width = width
        self.use_bigru = use_bigru
        self.use_transformer = use_transformer
        self.use_film = use_film
        self.use_gate = use_gate and (use_bigru and use_transformer)

        # Conv branches
        self.b1 = nn.Sequential(MultiKernelConvBlock(x1_channels, width), SEBlock(width))
        self.b2 = nn.Sequential(DilatedConvBlock(x2_channels, width),    SEBlock(width))
        self.posenc = PositionalEncoding1D(width, max_len=max_len)

        # ESM token projection
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_token_dim, width),
            nn.ReLU(inplace=True),
            nn.LayerNorm(width),
        )

        # Cross-attention: q from concat(x1, x2), kv from ESM.
        self.q_proj = nn.Linear(width * 2, width)
        self.cross_attn = CrossAttention(
            q_dim=width, kv_dim=width, out_dim=width, heads=4, dropout=dropout
        )

        # Optional FiLM conditioning from pooled ESM.
        if use_film:
            self.film1 = FiLM(film_cond_dim, width)
            self.film2 = FiLM(film_cond_dim, width)

        # Parallel sequence-model branches.
        if use_bigru:
            self.bigru = BiGRUBlockResLN(channels=width, hidden_size=width)
        if use_transformer:
            self.tr = TransformerBlock(d_model=width, nhead=4,
                                       dim_ff=width * 2, dropout=dropout)

        # Gate (only meaningful when both branches exist).
        if self.use_gate:
            self.gate = DualBranchGate(channels=width,
                                       init_bias=gate_init_bias,
                                       tau=gate_tau)

        # Classifier head: pool(x1) + pool(x2) + pool(fused)  Linear.
        self.classifier = nn.Sequential(
            nn.Linear(width * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # For analysis / logging only.
        self.last_gate_alpha = None

    # ---------------- helpers ----------------
    @staticmethod
    def _masked_pool(x, mask):
        """
        [M4] Masked avg + max pool. AMP-safe (no -inf comparisons).
        x    : [B, C, L]
        mask : [B, L] bool (True = valid) or None
        returns: [B, 2C]
        """
        if mask is None:
            return torch.cat([
                F.adaptive_avg_pool1d(x, 1).squeeze(-1),
                F.adaptive_max_pool1d(x, 1).squeeze(-1),
            ], dim=1)
        m = mask.unsqueeze(1).float()                # [B,1,L]
        denom = m.sum(-1).clamp_min(1e-6)            # [B,1]
        avg = (x * m).sum(-1) / denom                # [B,C]

        NEG = torch.finfo(x.dtype).min / 2           # AMP-safe negative
        x_masked = x.masked_fill(~m.bool(), NEG)
        mx = x_masked.max(-1).values
        # If an entire row is pad (shouldn't happen, but safe), use 0.
        any_valid = (m.sum(-1) > 0).expand_as(mx)    # [B,C]
        mx = torch.where(any_valid, mx, torch.zeros_like(mx))
        return torch.cat([avg, mx], dim=1)

    @staticmethod
    def _re_mask(x, mask):
        if mask is None:
            return x
        return x * mask.unsqueeze(1).float()

    def _seq_model(self, x, mask):
        """
        Apply BiGRU / Transformer (parallel with gate if both enabled).
        """
        has_gru = self.use_bigru
        has_tr  = self.use_transformer

        if has_gru and has_tr and self.use_gate:
            y_gru = self.bigru(x)
            y_gru = self._re_mask(y_gru, mask)        # [M3]
            y_tr  = self.tr(x, key_padding_mask=mask) # already masked inside
            fused, alpha = self.gate(y_gru, y_tr)
            self.last_gate_alpha = alpha.detach()
            fused = self._re_mask(fused, mask)
            return fused

        if has_gru and has_tr and not self.use_gate:
            # Fall back to series (V5 behaviour) for ablation.
            x = self.bigru(x)
            x = self._re_mask(x, mask)
            x = self.tr(x, key_padding_mask=mask)
            return x

        if has_gru:
            x = self.bigru(x)
            return self._re_mask(x, mask)

        if has_tr:
            return self.tr(x, key_padding_mask=mask)

        # neither: identity
        return x

    # ---------------- forward ----------------
    def forward(self, x1, x2, esm_token, esm_pooled=None, mask=None):
        # Conv + positional encoding.
        x1 = self.posenc(self.b1(x1))   # [B, W, L]
        x2 = self.posenc(self.b2(x2))   # [B, W, L]
        x1 = self._re_mask(x1, mask)
        x2 = self._re_mask(x2, mask)

        # FiLM + re-mask [M2].
        if self.use_film and (esm_pooled is not None):
            x1 = self.film1(x1, esm_pooled)
            x2 = self.film2(x2, esm_pooled)
            x1 = self._re_mask(x1, mask)
            x2 = self._re_mask(x2, mask)

        # ESM token projection.
        esm_t = self.esm_proj(esm_token)  # [B, L, W]
        if mask is not None:
            esm_t = esm_t * mask.unsqueeze(-1).float()

        # Cross-attention: q from concat(x1, x2).
        q = self.q_proj(torch.cat([x1, x2], dim=1).transpose(1, 2))  # [B, L, W]
        if mask is not None:
            q = q * mask.unsqueeze(-1).float()
        fused = self.cross_attn(q, k=esm_t, v=esm_t,
                                key_padding_mask=mask,
                                query_padding_mask=mask).transpose(1, 2)  # [B, W, L]

        # Sequence modelling (parallel gate or fallback).
        fused = self._seq_model(fused, mask)

        # Masked pools + classifier.
        g1 = self._masked_pool(x1, mask)
        g2 = self._masked_pool(x2, mask)
        gf = self._masked_pool(fused, mask)
        feat = torch.cat([g1, g2, gf], dim=1)
        return self.classifier(feat)
