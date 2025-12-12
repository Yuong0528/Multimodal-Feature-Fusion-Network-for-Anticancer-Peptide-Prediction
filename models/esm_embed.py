# import torch
# from esm import pretrained
#
# _model, _alphabet = pretrained.esm2_t33_650M_UR50D()  # 1280 dim
# _model.eval()
# _batch_converter = _alphabet.get_batch_converter()
#
# @torch.no_grad()
# def esm_embed_batch(seqs):
#     """
#     Input:list[str] any Length
#     Output:[B,1024]<cls> token
#     """
#     data = [(f"id{i}", seq) for i, seq in enumerate(seqs)]
#     _, _, tokens = _batch_converter(data)
#     results = _model(tokens, repr_layers=[33])
#     return results["representations"][33][:, 0, :]  # [B,1024]
# models/esm_embed.py
# import torch
# from esm import pretrained
# _model, _alphabet = pretrained.esm2_t33_650M_UR50D()
# _model.eval()
# _batch_converter = _alphabet.get_batch_converter()
#
# @torch.no_grad()
# def esm_embed_batch(seqs):
#
#     data = [(f"id{i}", seq) for i, seq in enumerate(seqs)]
#     _, _, tokens = _batch_converter(data)
#     results = _model(tokens, repr_layers=[33])
#     return results["representations"][33][:, 0, :]
#
#
# @torch.no_grad()
# def esm_embed_tokens(seqs):
#     """
#     ÔÞÏ* (tokens:[L,1280], pooled:[1280])
#     """
#     data = [(f"id{i}", seq) for i, seq in enumerate(seqs)]
#     _, _, tokens = _batch_converter(data)
#     results = _model(tokens, repr_layers=[33])
#     rep = results["representations"][33]  # [B,L,D]
#
#     out = []
#     for i, seq in enumerate(seqs):
#         L = len(seq)
#         token_repr = rep[i, 1:L+1].cpu()
#         pooled_repr = rep[i, 0].cpu()
#         out.append((token_repr, pooled_repr))
#     return out if len(out) > 1 else out[0]

# models/esm_embed.py
# -*- coding: utf-8 -*-
import torch
from esm import pretrained

_model, _alphabet = pretrained.esm2_t33_650M_UR50D()
_model.eval()
_batch_converter = _alphabet.get_batch_converter()

@torch.no_grad()
def esm_embed_batch(seqs):
    data = [(f"id{i}", s) for i, s in enumerate(seqs)]
    _, _, toks = _batch_converter(data)        # [B, L+2]
    out = _model(toks, repr_layers=[33])
    return out["representations"][33][:, 0, :] # [B, 1280]

@torch.no_grad()
def esm_embed_tokens(seqs):

    data = [(f"id{i}", s) for i, s in enumerate(seqs)]
    _, _, toks = _batch_converter(data)
    out = _model(toks, repr_layers=[33], return_contacts=False)
    rep = out["representations"][33]           # [B, L+2, 1280]

    result = []
    for i, s in enumerate(seqs):
        L = len(s)
        tokens = rep[i, 1:L+1, :].cpu()        # [L, 1280] 1..L» CLS/EOS
        pooled = rep[i, 0, :].cpu()            # [1280]    CLS
        result.append((tokens, pooled))
    return result if len(result) > 1 else result[0]
