import torch
from torch import nn
import math
import numpy as np
from inspect import signature


class MultiheadAttention(nn.Module):
    def __init__(self,
                 key_dim,
                 query_dim,
                 att_dim,
                 out_dim,
                 num_heads,
                 dropout,
                 add_bias):
        super().__init__()

        assert att_dim % num_heads == 0
        self.head_dim = att_dim // num_heads
        self.num_heads = num_heads
        self.scale = math.sqrt(self.head_dim)

        self.w_key = nn.Linear(key_dim, att_dim, bias=add_bias)
        self.w_value = nn.Linear(key_dim, att_dim, bias=add_bias)
        self.w_query = nn.Linear(query_dim, att_dim, bias=add_bias)

        self.w_out = nn.Linear(att_dim, out_dim, bias=add_bias)

        self.dropout_attn = nn.Dropout(p=dropout)

        self.neg_inf = None

    def reset_parameters(self, add_bias):
        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_out.weight)
        if add_bias:
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_value.bias, 0.)
            nn.init.constant_(self.w_query.bias, 0.)
            nn.init.constant_(self.w_out.bias, 0.)

    def forward(self,
                key: torch.Tensor,
                value: torch.Tensor,
                query: torch.Tensor,
                mask: torch.Tensor):
        if self.neg_inf is None:
            self.neg_inf = float(np.finfo(
                torch.tensor(0, dtype=key.dtype).numpy().dtype).min)

        bsize, klen = key.size()[: 2]
        qlen = query.size(1)

        # key: `[N, klen, kdim] -> [N, klen, adim] -> [N, klen, H, adim/H(=hdim)]`
        # value: `[N, klen, vdim] -> [N, klen, adim] -> [N, klen, H, adim/H(=hdim)]`
        # query: `[N, qlen, qdim] -> [N, qlen, adim] -> [N, qlen, H, adim/H(=hdim)]`
        key = self.w_key(key).reshape([bsize, -1, self.num_heads, self.head_dim])
        value = self.w_value(value).reshape([bsize, -1, self.num_heads, self.head_dim])
        query = self.w_query(query).reshape([bsize, -1, self.num_heads, self.head_dim])

        # qk_score: `[N, qlen, H, hdim] x [N, klen, H, hdim] -> [N, qlen, klen, H]`
        qk_score = torch.einsum("bihd,bjhd->bijh", (query, key)) / self.scale

        # Apply mask.
        if mask is not None:
            # `[N, qlen, klen] -> [N, qlen, klen, H]`
            mask = mask.unsqueeze(3).repeat([1, 1, 1, self.num_heads])
            mask_size = (bsize, qlen, klen, self.num_heads)
            assert mask.size() == mask_size, f"{mask.size()}:{mask_size}"
            # Negative infinity should be 0 in softmax.
            qk_score = qk_score.masked_fill_(mask == 0, self.neg_inf)
        # Compute attention weight.
        attw = torch.softmax(qk_score, dim=2)
        attw = self.dropout_attn(attw)

        # cvec: `[N, qlen, klen, H] x [N, qlen, H, hdim] -> [N, qlen, H, hdim]
        # -> [N, qlen, H * hdim]`
        cvec = torch.einsum("bijh,bjhd->bihd", (attw, value))
        cvec = cvec.reshape([bsize, -1, self.num_heads * self.head_dim])
        cvec = self.w_out(cvec)
        # attw: `[N, qlen, klen, H]` -> `[N, H, qlen, klen]`
        attw = attw.permute(0, 3, 1, 2)
        return cvec, attw

class PositionwiseFeedForward(nn.Module):
    def __init__(self,
                 dim_model,
                 dim_ffw,
                 dropout,
                 activation,
                 add_bias):
       super().__init__()

       self.w_1 = nn.Linear(dim_model, dim_ffw, bias=add_bias)
       self.w_2 = nn.Linear(dim_ffw, dim_model, bias=add_bias)

       self.dropout = nn.Dropout(p=dropout)

       self.activation = select_reluwise_activation(activation)

    def forward(self, feature):
        feature = self.w_1(feature)
        feature = self.activation(feature)
        feature = self.dropout(feature)
        feature = self.w_2(feature)
        return feature

def make_san_mask(pad_mask,
                  causal_mask):
    # xx_mask: `[N, qlen, klen]`
    san_mask = pad_mask.unsqueeze(1).repeat([1, pad_mask.shape[-1], 1])
    if causal_mask is not None:
        san_mask = san_mask & causal_mask
    return san_mask

def select_reluwise_activation(activation):
    if activation == "relu":
        layer = torch.nn.ReLU()
    elif activation == "gelu":
        layer = torch.nn.GELU()
    elif activation in ["swish", "silu"]:
        layer = torch.nn.SILU()
    elif activation == "mish":
        layer = torch.nn.Mish()
    else:
        raise NotImplementedError(f"Activation for {activation} is not implemented.")
    return layer

def create_norm(norm_type, dim_model, eps=1e-5, add_bias=None):
    # The argument `bias` was added at v2.1.0.
    # So, we check whether LayerNorm has this.
    sig = signature(nn.LayerNorm)
    available_bias = bool("bias" in sig.parameters)
    if norm_type == "layer":
        if available_bias:
            norm = nn.LayerNorm(dim_model, eps=eps, bias=add_bias)
        else:
            norm = nn.LayerNorm(dim_model, eps=eps)
    elif norm_type == "batch":
        norm = nn.BatchNorm1d(dim_model, eps=eps)
    elif norm_type == "batch2d":
        norm = nn.BatchNorm2d(dim_model, eps=eps)
    return norm

def apply_norm(norm_layer, feature, channel_first=False):
    shape = feature.shape
    if len(shape) > 5:
        raise NotImplementedError("Unsupported feature shape:{shape}")
    if isinstance(norm_layer, nn.LayerNorm):
        if channel_first:
            if len(shape) == 3:
                # `[N, C, *] -> [N, *, C] -> [N, C, *]`
                feature = feature.permute([0, 2, 1]).contiguous()
                feature = norm_layer(feature)
                feature = feature.permute([0, 2, 1]).contiguous()
            elif len(shape) == 4:
                # `[N, C, *, *] -> [N, *, *, C] -> [N, C, *, *]`
                feature = feature.permute([0, 2, 3, 1]).contiguous()
                feature = norm_layer(feature)
                feature = feature.permute([0, 3, 1, 2]).contiguous()
            elif len(shape) == 5:
                # `[N, C, *, *, *] -> [N, *, *, *, C] -> [N, C, *, *, *]`
                feature = feature.permute([0, 2, 3, 4, 1]).contiguous()
                feature = norm_layer(feature)
                feature = feature.permute([0, 4, 1, 2, 3]).contiguous()
        else:
            feature = norm_layer(feature)
    elif isinstance(norm_layer, nn.BatchNorm1d):
        if channel_first:
            feature = norm_layer(feature)
        else:
            if len(shape) == 3:
                # `[N, *, C] -> [N, C, *] -> [N, *, C]`
                feature = feature.permute([0, 2, 1]).contiguous()
                feature = norm_layer(feature)
                feature = feature.permute([0, 2, 1]).contiguous()
            elif len(shape) == 4:
                # `[N, *, *, C] -> [N, C, *, *] -> [N, *, *, C]`
                feature = feature.permute([0, 3, 1, 2]).contiguous()
                feature = norm_layer(feature)
                feature = feature.permute([0, 2, 3, 1]).contiguous()
            elif len(shape) == 5:
                # `[N, *, *, C] -> [N, C, *, *] -> [N, *, *, C]`
                feature = feature.permute([0, 4, 1, 2, 3]).contiguous()
                feature = norm_layer(feature)
                feature = feature.permute([0, 2, 3, 4, 1]).contiguous()
    return feature