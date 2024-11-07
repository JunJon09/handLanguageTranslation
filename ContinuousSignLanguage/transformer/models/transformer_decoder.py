import transformer.models.transformer_modules as modules
from torch import nn
import torch



class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 dim_model,
                 num_heads,
                 dim_ffw,
                 dropout,
                 activation,
                 norm_type_sattn,
                 norm_type_cattn,
                 norm_type_ffw,
                 norm_eps,
                 norm_first,
                 add_bias):
        super().__init__()

        self.norm_first = norm_first

        #################################################
        # MHSA.
        #################################################
        self.self_attn = modules.MultiheadAttention(
            key_dim=dim_model,
            query_dim=dim_model,
            att_dim=dim_model,
            out_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            add_bias=add_bias)
        self.norm_sattn = modules.create_norm(norm_type_sattn, dim_model, norm_eps, add_bias)

        #################################################
        # MHCA.
        #################################################
        self.cross_attn = modules.MultiheadAttention(
            key_dim=dim_model,
            query_dim=dim_model,
            att_dim=dim_model,
            out_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            add_bias=add_bias)
        self.norm_cattn = modules.create_norm(norm_type_cattn, dim_model, norm_eps, add_bias)

        #################################################
        # PFFN.
        #################################################
        self.ffw = modules.PositionwiseFeedForward(
            dim_model=dim_model,
            dim_ffw=dim_ffw,
            dropout=dropout,
            activation=activation,
            add_bias=add_bias)
        self.norm_ffw = modules.create_norm(norm_type_ffw, dim_model, norm_eps, add_bias)

        self.dropout = nn.Dropout(p=dropout)

        # To store attention weights.
        self.sattw = None
        self.cattw = None

    def _forward_prenorm(self,
                         tgt_feature,
                         enc_feature,
                         tgt_san_mask,
                         enc_tgt_mask):
        """Pre-normalization structure.

        For the details, please refer
        https://arxiv.org/pdf/2002.04745v1.pdf
        """
        #################################################
        # self-attention
        #################################################
        residual = tgt_feature
        tgt_feature = modules.apply_norm(self.norm_sattn, tgt_feature)
        tgt_feature, self.sattw = self.self_attn(
            key=tgt_feature,
            value=tgt_feature,
            query=tgt_feature,
            mask=tgt_san_mask)
        tgt_feature = self.dropout(tgt_feature) + residual

        #################################################
        # cross-attention
        #################################################
        residual = tgt_feature
        tgt_feature = modules.apply_norm(self.norm_cattn, tgt_feature)
        tgt_feature, self.cattw = self.cross_attn(
            key=enc_feature,
            value=enc_feature,
            query=tgt_feature,
            mask=enc_tgt_mask)
        tgt_feature = self.dropout(tgt_feature) + residual

        #################################################
        # FFW
        #################################################
        residual = tgt_feature
        tgt_feature = modules.apply_norm(self.norm_ffw, tgt_feature)
        tgt_feature = self.ffw(tgt_feature)
        tgt_feature = self.dropout(tgt_feature) + residual
        return tgt_feature

    def _forward_postnorm(self,
                          tgt_feature,
                          enc_feature,
                          tgt_san_mask,
                          enc_tgt_mask):
        """Post-normalization structure (standard).

        """
        #################################################
        # self-attention
        #################################################
        residual = tgt_feature
        tgt_feature, self.sattw = self.self_attn(
            key=tgt_feature,
            value=tgt_feature,
            query=tgt_feature,
            mask=tgt_san_mask)
        tgt_feature = self.dropout(tgt_feature) + residual
        tgt_feature = modules.apply_norm(self.norm_sattn, tgt_feature)

        #################################################
        # cross-attention
        #################################################
        residual = tgt_feature
        tgt_feature, self.cattw = self.cross_attn(
            key=enc_feature,
            value=enc_feature,
            query=tgt_feature,
            mask=enc_tgt_mask)
        tgt_feature = self.dropout(tgt_feature) + residual
        tgt_feature = modules.apply_norm(self.norm_cattn, tgt_feature)

        #################################################
        # FFW
        #################################################
        residual = tgt_feature
        tgt_feature = self.ffw(tgt_feature)
        tgt_feature = self.dropout(tgt_feature) + residual
        tgt_feature = modules.apply_norm(self.norm_ffw, tgt_feature)

        return tgt_feature

    def forward(self,
                tgt_feature,
                enc_feature,
                tgt_causal_mask=None,
                enc_tgt_causal_mask=None,
                tgt_key_padding_mask=None,
                enc_key_padding_mask=None):

        # Create mask.
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = torch.ones(tgt_feature.shape[:2],
                                              dtype=enc_feature.dtype,
                                              device=enc_feature.device)
        tgt_san_mask = modules.make_san_mask(tgt_key_padding_mask, tgt_causal_mask)
        if enc_key_padding_mask is None:
            enc_key_padding_mask = torch.ones(enc_feature.shape[:2],
                                              dtype=enc_feature.dtype,
                                              device=enc_feature.device)
        enc_tgt_mask = enc_key_padding_mask.unsqueeze(1).repeat(
            [1, tgt_feature.shape[1], 1])
        if enc_tgt_causal_mask is not None:
            enc_tgt_mask = enc_tgt_mask & enc_tgt_causal_mask

        if self.norm_first:
            tgt_feature = self._forward_prenorm(tgt_feature, enc_feature,
                                                tgt_san_mask, enc_tgt_mask)
        else:
            tgt_feature = self._forward_postnorm(tgt_feature, enc_feature,
                                                 tgt_san_mask, enc_tgt_mask)

        return tgt_feature