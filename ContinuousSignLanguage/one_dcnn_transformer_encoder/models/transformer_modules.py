import torch
from torch import nn
import math
import numpy as np
from inspect import signature
from pydantic import (
    BaseModel,
    ConfigDict,
    Field)

class PositionalEncoding(nn.Module):
    def __init__(self,
                 dim_model: int,
                 dropout: float,
                 max_len: int = 5000):
        super().__init__()
        self.dim_model = dim_model
        # Compute the positional encodings once in log space.
        pose = torch.zeros(max_len, dim_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float()
                             * -(math.log(10000.0) / dim_model))
        pose[:, 0::2] = torch.sin(position * div_term)
        pose[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pose", pose)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                feature):
        feature = feature + self.pose[None, :feature.shape[1], :]
        feature = self.dropout(feature)
        return feature

class MultiheadAttention(nn.Module):
    """Multi-headed attention (MHA) layer.

    # Args:
      - key_dim: The dimension of key.
      - query_dim: The dimension of query.
      - att_dim: The dimension of attention space.
      - out_dim: The dimension of output.
      - num_heads: The number of heads.
      - dropout: The dropout probability for attention weights.
      - add_bias: If True, use bias term in linear layers.
    """
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

        self.qkv_same_dim = key_dim == query_dim
        self.reset_parameters(add_bias)

    def reset_parameters(self, add_bias):
        """Initialize parameters with Xavier uniform distribution.

        # NOTE: For this initialization, please refer
        https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py  # pylint: disable=line-too-long

        """
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.w_key.weight)
            nn.init.xavier_uniform_(self.w_value.weight)
            nn.init.xavier_uniform_(self.w_query.weight)
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
        """Perform forward computation.

        # Args:
          - key: `[N, klen, key_dim]`
          - value: `[N, klen, vdim]`
          - query: `[N, qlen, query_dim]`
          - mask: `[N, qlen, klen]`
        # Returns:
          - cvec: The context vector. `[N, qlen, vdim]`
          - aws: The attention weights. `[N, H, qlen, klen]`
        """
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

        # cvec: `[N, qlen, klen, H] x [N, qlen, h, hdim] -> [N, qlen, H, hdim]
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
    return norm


def apply_norm(norm_layer, feature, channel_first=False):
    if isinstance(norm_layer, nn.LayerNorm):
        if channel_first:
            # `[N, C, T] -> [N, T, C] -> [N, C, T]`
            feature = feature.permute([0, 2, 1]).contiguous()
            feature = norm_layer(feature)
            feature = feature.permute([0, 2, 1]).contiguous()
        else:
            feature = norm_layer(feature)
    elif isinstance(norm_layer, nn.BatchNorm1d):
        if channel_first:
            feature = norm_layer(feature)
        else:
            # `[N, T, C] -> [N, C, T]`
            feature = feature.permute([0, 2, 1]).contiguous()
            feature = norm_layer(feature)
            # `[N, C, T] -> [N, T, C]`
            feature = feature.permute([0, 2, 1]).contiguous()
    return feature



def select_reluwise_activation(activation):
    if activation == "relu":
        layer = torch.nn.ReLU()
    elif activation == "gelu":
        layer = torch.nn.GELU()
    elif activation in ["swish", "silu"]:
        layer = torch.nn.SiLU()
    elif activation == "mish":
        layer = torch.nn.Mish()
    elif activation == "geluacc":
        layer = GELUAcc()
    elif activation == "tanhexp":
        layer = TanhExp()
    else:
        raise NotImplementedError(f"Activation for {activation} is not implemented.")
    return layer



def make_san_mask(pad_mask,
                  causal_mask):
    """Make self-attention mask.

    # Args:
      - pad_mask: The padding mask. `[N, T]`
      - causal_mask: `[N, T (query), T (key)]`
        The mask for future context. For example, if T = 3, causal_mask
        should be:
        [[1, 0, 0,
         [1, 1, 0,
         [1, 1, 1]]
    # Returns:
      - san_mask: `[N, T (query), T (key)]`
    """
    # xx_mask: `[N, qlen, klen]`
    san_mask = pad_mask.unsqueeze(1).repeat([1, pad_mask.shape[-1], 1])
    if causal_mask is not None:
        san_mask = san_mask & causal_mask
    return san_mask

def create_encoder_mask(src_key_padding_mask,
                        causal_mask):
    if src_key_padding_mask is not None:
        san_mask = make_san_mask(src_key_padding_mask, causal_mask)
    elif causal_mask is not None:
        san_mask = causal_mask
    else:
        san_mask = None
    return san_mask

class FuncTanhExp(torch.autograd.Function):  # pylint: disable=W0223
    """Implementation of TanhExp activation.
    """
    # It is difficult to handle arguments-differ correctly.
    # https://github.com/PyCQA/pylint/issues/3812
    # pylint: disable=arguments-differ
    @staticmethod
    def forward(ctx, inputs):
        """Perform foward computation."""
        inputs = torch.clamp(inputs, max=88)
        ctx.save_for_backward(inputs)
        output = inputs * torch.tanh(torch.exp(inputs))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Perform backward computation."""
        inputs, = ctx.saved_tensors
        results = grad_output * (torch.tanh(torch.exp(inputs))
                                 + inputs * (1 - torch.tanh(torch.exp(inputs))**2)
                                 * torch.exp(inputs))
        return results


class TanhExp(nn.Module):
    """Applies TanhExp function.

    # Note
      For the details, please see https://arxiv.org/abs/2003.09855 .
    """
    def __init__(self):
        # Keep for the consistent style.
        # pylint: disable=useless-super-delegation
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform foward computation."""
        return FuncTanhExp.apply(inputs)


class GELUAcc(nn.Module):
    """Accurate approximation of GELU function.

    https://github.com/hendrycks/GELUs

    """
    FACTOR = math.sqrt(2 / math.pi)

    def __init__(self):
        # Keep for the consistent style.
        # pylint: disable=useless-super-delegation
        super().__init__()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Perform foward computation."""
        return 0.5 * feature * (1 + torch.tanh(self.FACTOR * (
            feature + 0.044715 * torch.pow(feature, 3))))

class Identity(nn.Module):
    """Place holder layer to return identity vector.
    """
    # This design is on purpose.
    # pylint: disable=unused-argument
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feature, *args, **kwargs):
        """Perform forward computation.
        """
        return feature