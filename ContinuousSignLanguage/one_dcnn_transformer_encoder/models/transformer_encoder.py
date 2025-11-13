import cnn_transformer.models.transformer_modules as modules
from torch import nn
import copy

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 dim_model,
                 num_heads,
                 dim_ffw,
                 dropout,
                 activation,
                 norm_type_sattn,
                 norm_type_ffw,
                 norm_eps,
                 norm_first,
                 add_bias):
        super().__init__()

        self.norm_first = norm_first

        #################################################
        # MHA.
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
        self.attw = None

    def _forward_prenorm(self,
                         feature,
                         san_mask):
        """Pre-normalization structure.

        For the details, please refer
        https://arxiv.org/pdf/2002.04745v1.pdf
        """
        #################################################
        # self-attention
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = modules.apply_norm(self.norm_sattn, feature)
        feature, self.attw = self.self_attn(
            key=feature,
            value=feature,
            query=feature,
            mask=san_mask)
        feature = self.dropout(feature) + residual

        #################################################
        # FFW
        #################################################
        residual = feature
        # `[N, qlen, dim_model]`
        feature = modules.apply_norm(self.norm_ffw, feature)
        feature = self.ffw(feature)
        feature = self.dropout(feature) + residual
        return feature

    def _forward_postnorm(self,
                          feature,
                          san_mask):
        """Post-normalization structure (standard).

        """
        #################################################
        # self-attention
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature, self.attw = self.self_attn(
            key=feature,
            value=feature,
            query=feature,
            mask=san_mask)
        feature = self.dropout(feature) + residual
        feature = modules.apply_norm(self.norm_sattn, feature)

        #################################################
        # FFW
        #################################################
        residual = feature
        # `[N, qlen, dim_model]`
        feature = self.ffw(feature)
        feature = self.dropout(feature) + residual
        feature = modules.apply_norm(self.norm_ffw, feature)
        return feature

    def forward(self,
                feature,
                causal_mask=None,
                src_key_padding_mask=None):
        bsize, qlen = feature.shape[:2]
        if src_key_padding_mask is not None:
            san_mask = modules.make_san_mask(src_key_padding_mask, causal_mask)
        elif causal_mask is not None:
            san_mask = causal_mask
        else:
            san_mask = None

        if self.norm_first:
            feature = self._forward_prenorm(feature, san_mask)
        else:
            feature = self._forward_postnorm(feature, san_mask)

        return feature


class TransformerEncoder(nn.Module):
    def __init__(self,
                 encoder_layer,
                 num_layers,
                 dim_model,
                 dropout_pe,
                 norm_type_tail,
                 norm_eps,
                 norm_first,
                 add_bias,
                 add_tailnorm):
        super().__init__()

        self.pos_encoder = modules.PositionalEncoding(dim_model, dropout_pe)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

        # Add LayerNorm at tail position.
        # This is applied only when norm_first is True because
        # post-normalization structure includes tail-normalization in encoder
        # layers.
        if add_tailnorm and norm_first:
            self.norm_tail = modules.create_norm(norm_type_tail, dim_model, norm_eps, add_bias)
        else:
            self.norm_tail = modules.Identity()

    def forward(self,
                feature,
                causal_mask,
                src_key_padding_mask):
        feature = self.pos_encoder(feature)
        for layer in self.layers:
            feature = layer(feature,
                            causal_mask,
                            src_key_padding_mask)
        feature = modules.apply_norm(self.norm_tail, feature)
        return feature