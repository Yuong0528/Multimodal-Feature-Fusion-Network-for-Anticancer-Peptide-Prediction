# models/base_model.py
# -*- coding: utf-8 -*-
import math, torch
import torch.nn as nn
import torch.nn.functional as F

# ============== Wavelet (optional) ==============
try:
    from pytorch_wavelets import DWT1D, IDWT1D
    _HAS_WAVELET = True
except Exception:
    _HAS_WAVELET = False


class WaveletDenoise1D(nn.Module):
    def __init__(self, wavelet='db1', level=1, threshold=0.1):
        super().__init__()
        assert _HAS_WAVELET, "pytorch_wavelets not installed!"
        self.dwt = DWT1D(wave=wavelet, J=level, mode='symmetric')
        self.idwt = IDWT1D(wave=wavelet, mode='symmetric')
        self.threshold = threshold

    def forward(self, x):  # x:[B,C,L]
        B, C, L = x.shape
        low, highs = self.dwt(x)
        denoised_highs = [
            torch.where(torch.abs(h) < self.threshold, torch.zeros_like(h),
                        h - torch.sign(h) * self.threshold)
            for h in highs
        ]
        rec = self.idwt((low, denoised_highs))
        return rec[..., :L] if rec.shape[-1] > L else F.pad(rec, (0, L - rec.shape[-1]))


# ============== basic blocks ==============
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # [B,C,L]
        scale = self.fc(x).unsqueeze(-1)
        return x * scale.expand_as(x)


class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):  # [B,C,L]
        return self.conv(x)


class MultiKernelConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):  # [B,C,L]
        return self.conv(x)


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # x:[B,C,L]
        B, C, L = x.shape
        return x + self.pe[:L, :C].T.unsqueeze(0)


# ============== sequence modeling ==============
class BiGRUBlockResLN(nn.Module):

    def __init__(self, channels, hidden_size):
        super().__init__()
        self.gru = nn.GRU(channels, hidden_size, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_size * 2, channels)
        self.ln   = nn.LayerNorm(channels)

    def forward(self, x):  # x:[B,C,L]
        x_l = x.transpose(1, 2)              # [B,L,C]
        y, _ = self.gru(x_l)                 # [B,L,2H]
        y = self.proj(y)                     # [B,L,C]
        y = self.ln(y + x_l)                 # î + LN
        return y.transpose(1, 2)             # [B,C,L]


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4, dim_ff=128, dropout=0.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)

    def forward(self, x):  # [B,C,L]
        x_l = x.transpose(1, 2)      # [B,L,C]
        attn, _ = self.sa(x_l, x_l, x_l)
        x_l = self.ln1(x_l + self.do(attn))
        ff = self.ff(x_l)
        x_l = self.ln2(x_l + self.do(ff))
        return x_l.transpose(1, 2)   # [B,C,L]


# ============== Cross-Attention & FiLM ==============
class CrossAttention(nn.Module):

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

    def forward(self, q, k, v, key_padding_mask=None):  # q,k,v: [B,L,C]
        B, L, C = q.shape
        H = self.heads
        qh = self.q_proj(q).view(B, L, H, C//H).transpose(1, 2)  # [B,H,L,C/H]
        kh = self.k_proj(k).view(B, L, H, C//H).transpose(1, 2)
        vh = self.v_proj(v).view(B, L, H, C//H).transpose(1, 2)

        attn = (qh @ kh.transpose(-2, -1)) * self.scale            # [B,H,L,L]
        if key_padding_mask is not None:
            pad = (~key_padding_mask).unsqueeze(1).unsqueeze(2)    # [B,1,1,L], True=pad
            attn = attn.masked_fill(pad, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = (attn @ vh).transpose(1, 2).contiguous().view(B, L, C)  # [B,L,C]
        out = self.o_proj(self.do(out))
        return self.ln(out + q)


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_channels):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_channels)
        self.beta  = nn.Linear(cond_dim, feat_channels)

    def forward(self, x, cond):  # x:[B,C,L], cond:[B,cond_dim]
        g = self.gamma(cond).unsqueeze(-1)  # [B,C,1]
        b = self.beta(cond).unsqueeze(-1)
        return x * (1.0 + g) + b


# ============== Model ==============
class EnhancedMFA_ACP(nn.Module):

    def __init__(
        self,
        x1_channels=20, x2_channels=5, esm_token_dim=1280,
        use_wavelet_on_x1=False, use_wavelet_on_x2=False,
        width=64, use_bigru=True, use_transformer=True,
        use_film=True, film_cond_dim=1280,
        max_len=512, dropout=0.2,
    ):
        super().__init__()
        self.width = width
        self.use_bigru = use_bigru
        self.use_transformer = use_transformer
        self.use_film = use_film

        # Wavelet
        self.wv1 = WaveletDenoise1D() if (use_wavelet_on_x1 and _HAS_WAVELET) else None
        self.wv2 = WaveletDenoise1D() if (use_wavelet_on_x2 and _HAS_WAVELET) else None

        # Conv branches
        self.b1 = nn.Sequential(MultiKernelConvBlock(x1_channels, width), SEBlock(width))
        self.b2 = nn.Sequential(DilatedConvBlock(x2_channels, width), SEBlock(width))
        self.posenc = PositionalEncoding1D(width, max_len=max_len)

        # ESM token -> width
        self.esm_proj = nn.Sequential(nn.Linear(esm_token_dim, width), nn.ReLU(inplace=True), nn.LayerNorm(width))

        self.q_proj = nn.Linear(width * 2, width)
        self.cross_attn = CrossAttention(q_dim=width, kv_dim=width, out_dim=width, heads=4, dropout=dropout)

        # Optional FiLM by pooled ESM
        if use_film:
            self.film1 = FiLM(film_cond_dim, width)
            self.film2 = FiLM(film_cond_dim, width)

        # Sequence modeling after fusion
        if use_bigru:
            self.bigru = BiGRUBlockResLN(channels=width, hidden_size=64)
        if use_transformer:
            self.tr = TransformerBlock(d_model=width, nhead=4, dim_ff=width*2, dropout=dropout)

        # Classifier head (pool x1/x2/fused -> concat)
        self.classifier = nn.Sequential(
            nn.Linear(width * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    @staticmethod
    def _masked_pool(x, mask):  # x:[B,C,L], mask:[B,L] (True=valid)
        if mask is None:
            return torch.cat([
                F.adaptive_avg_pool1d(x, 1).squeeze(-1),
                F.adaptive_max_pool1d(x, 1).squeeze(-1)
            ], dim=1)
        m = mask.unsqueeze(1).float()                 # [B,1,L]
        # avg
        sumv = (x * m).sum(-1)
        denom = m.sum(-1).clamp_min(1e-6)
        avg = sumv / denom
        # max
        x_masked = x.masked_fill(~m.bool(), float('-inf'))
        mx = x_masked.max(-1).values
        mx[mx == float('-inf')] = 0.0
        return torch.cat([avg, mx], dim=1)

    def forward(self, x1, x2, esm_token, esm_pooled=None, mask=None):
        # wavelet
        if self.wv1 is not None: x1 = self.wv1(x1)
        if self.wv2 is not None: x2 = self.wv2(x2)

        # conv + posenc
        x1 = self.posenc(self.b1(x1))   # [B,W,L]
        x2 = self.posenc(self.b2(x2))   # [B,W,L]

        if mask is not None:
            m = mask.unsqueeze(1).float()
            x1, x2 = x1 * m, x2 * m

        # FiLM
        if self.use_film and (esm_pooled is not None):
            x1 = self.film1(x1, esm_pooled)
            x2 = self.film2(x2, esm_pooled)

        # ESM per-residue -> width
        esm_t = self.esm_proj(esm_token)           # [B,L,W]
        if mask is not None:
            esm_t = esm_t * mask.unsqueeze(-1).float()

        # Cross-Attn (q from convs; k=v from ESM tokens)
        q = self.q_proj(torch.cat([x1, x2], dim=1).transpose(1, 2))  # [B,L,W]
        if mask is not None:
            q = q * mask.unsqueeze(-1).float()

        fused = self.cross_attn(q, k=esm_t, v=esm_t, key_padding_mask=mask).transpose(1, 2)  # [B,W,L]

        # sequence modeling after fusion
        if self.use_bigru:
            fused = self.bigru(fused)         # [B,W,L]
        if self.use_transformer:
            fused = self.tr(fused)            # [B,W,L]

        # masked pools
        g1 = self._masked_pool(x1, mask)      # [B,2W]
        g2 = self._masked_pool(x2, mask)      # [B,2W]
        gf = self._masked_pool(fused, mask)   # [B,2W]

        feat = torch.cat([g1, g2, gf], dim=1) # [B,6W]
        return self.classifier(feat)# [B,1]



class EnhancedMFA_ACP_V5_ESMOnly(EnhancedMFA_ACP_V5):
    def forward(self, x1=None, x2=None, esm_token=None, esm_pooled=None, mask=None):
        assert esm_token is not None, "ESM token must be provided"

        B, L, _ = esm_token.shape
        W = self.width

        # ESM per-residue -> width
        esm_t = self.esm_proj(esm_token)  # [B,L,W]
        if mask is not None:
            esm_t = esm_t * mask.unsqueeze(-1).float()

        x_fused = esm_t.transpose(1, 2)  #
        if self.use_film and (esm_pooled is not None):
            x_fused = self.film1(x_fused, esm_pooled)

        # sequence modeling
        if self.use_bigru:
            x_fused = self.bigru(x_fused)
        if self.use_transformer:
            x_fused = self.tr(x_fused)

        # masked pooling
        gf = self._masked_pool(x_fused, mask)  # [B,2W]

        zeros = torch.zeros_like(gf)
        feat = torch.cat([zeros, zeros, gf], dim=1)  # [B,6W]

        return self.classifier(feat)  # [B,1]







class EnhancedMFA_ACP_V5_ESMOnly(EnhancedMFA_ACP_V5):
    def forward(self, x1=None, x2=None, esm_token=None, esm_pooled=None, mask=None):
        assert esm_token is not None, "ESM token must be provided"

        B, L, _ = esm_token.shape
        W = self.width

        # ESM per-residue -> width
        esm_t = self.esm_proj(esm_token)  # [B,L,W]
        if mask is not None:
            esm_t = esm_t * mask.unsqueeze(-1).float()

        x_fused = esm_t.transpose(1, 2)  #
        if self.use_film and (esm_pooled is not None):
            x_fused = self.film1(x_fused, esm_pooled)

        # sequence modeling
        if self.use_bigru:
            x_fused = self.bigru(x_fused)
        if self.use_transformer:
            x_fused = self.tr(x_fused)

        # masked pooling
        gf = self._masked_pool(x_fused, mask)  # [B,2W]

        zeros = torch.zeros_like(gf)
        feat = torch.cat([zeros, zeros, gf], dim=1)  # [B,6W]

        return self.classifier(feat)  # [B,1]



class EnhancedMFA_ACP_V5_X1Only(EnhancedMFA_ACP_V5):
    def forward(self, x1=None, x2=None, esm_token=None, esm_pooled=None, mask=None):
        assert x1 is not None, "x1 input must be provided"
        W = self.width

        x1_proc = self.posenc(self.b1(x1))
        if mask is not None:
            x1_proc = x1_proc * mask.unsqueeze(1).float()


        if self.use_film and (esm_pooled is not None):
            x1_proc = self.film1(x1_proc, esm_pooled)

        g1 = self._masked_pool(x1_proc, mask)
        zeros = torch.zeros_like(g1)
        feat = torch.cat([g1, zeros, zeros], dim=1)  # [B,6W]
        return self.classifier(feat)

class EnhancedMFA_ACP_V5_X2Only(EnhancedMFA_ACP_V5):
    def forward(self, x1=None, x2=None, esm_token=None, esm_pooled=None, mask=None):
        assert x2 is not None, "x2 input must be provided"
        W = self.width

        x2_proc = self.posenc(self.b2(x2))
        if mask is not None:
            x2_proc = x2_proc * mask.unsqueeze(1).float()

        if self.use_film and (esm_pooled is not None):
            x2_proc = self.film2(x2_proc, esm_pooled)

        g2 = self._masked_pool(x2_proc, mask)
        zeros = torch.zeros_like(g2)
        feat = torch.cat([zeros, g2, zeros], dim=1)  # [B,6W]
        return self.classifier(feat)
#--------Analysis---------============++++++++#






# --------- x1 + x2 ---------
class EnhancedMFA_ACP_V5_X1X2(EnhancedMFA_ACP_V5):
    def forward(self, x1=None, x2=None, esm_token=None, esm_pooled=None, mask=None):
        assert x1 is not None and x2 is not None, "x1 and x2 inputs must be provided"
        W = self.width

        x1_proc = self.posenc(self.b1(x1))
        x2_proc = self.posenc(self.b2(x2))
        if mask is not None:
            m = mask.unsqueeze(1).float()
            x1_proc, x2_proc = x1_proc*m, x2_proc*m

        if self.use_film and (esm_pooled is not None):
            x1_proc = self.film1(x1_proc, esm_pooled)
            x2_proc = self.film2(x2_proc, esm_pooled)

        g1 = self._masked_pool(x1_proc, mask)
        g2 = self._masked_pool(x2_proc, mask)
        zeros = torch.zeros_like(g1)
        feat = torch.cat([g1, g2, zeros], dim=1)  # [B,6W]
        return self.classifier(feat)

# --------- x1 + ESM ---------
class EnhancedMFA_ACP_V5_X1ESM(EnhancedMFA_ACP_V5):
    def forward(self, x1=None, x2=None, esm_token=None, esm_pooled=None, mask=None):
        assert x1 is not None and esm_token is not None, "x1 and esm_token must be provided"
        W = self.width

        # x1
        x1_proc = self.posenc(self.b1(x1))
        if mask is not None:
            x1_proc = x1_proc * mask.unsqueeze(1).float()
        if self.use_film and (esm_pooled is not None):
            x1_proc = self.film1(x1_proc, esm_pooled)
        g1 = self._masked_pool(x1_proc, mask)

        # ESM
        esm_t = self.esm_proj(esm_token).transpose(1,2)  # [B,W,L]
        if mask is not None:
            esm_t = esm_t * mask.unsqueeze(1).float()
        if self.use_bigru:
            esm_t = self.bigru(esm_t)
        if self.use_transformer:
            esm_t = self.tr(esm_t)
        gf = self._masked_pool(esm_t, mask)

        zeros = torch.zeros_like(g1)
        feat = torch.cat([g1, zeros, gf], dim=1)  # [B,6W]
        return self.classifier(feat)
# --------- x2 + ESM ---------
class EnhancedMFA_ACP_V5_X2ESM(EnhancedMFA_ACP_V5):
    def forward(self, x1=None, x2=None, esm_token=None, esm_pooled=None, mask=None):
        assert x2 is not None and esm_token is not None, "x2 and esm_token must be provided"
        W = self.width

        # x2
        x2_proc = self.posenc(self.b2(x2))
        if mask is not None:
            x2_proc = x2_proc * mask.unsqueeze(1).float()
        if self.use_film and (esm_pooled is not None):
            x2_proc = self.film2(x2_proc, esm_pooled)
        g2 = self._masked_pool(x2_proc, mask)

        # ESM
        esm_t = self.esm_proj(esm_token).transpose(1,2)  # [B,W,L]
        if mask is not None:
            esm_t = esm_t * mask.unsqueeze(1).float()
        if self.use_bigru:
            esm_t = self.bigru(esm_t)
        if self.use_transformer:
            esm_t = self.tr(esm_t)
        gf = self._masked_pool(esm_t, mask)

        zeros = torch.zeros_like(g2)
        feat = torch.cat([zeros, g2, gf], dim=1)  # [B,6W]
        return self.classifier(feat)
