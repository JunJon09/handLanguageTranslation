import transformer.models.transformer_encoder as encoder
from torch import nn
import torch
from pydantic import (
    BaseModel,
    ConfigDict)
import math
from torch.nn import functional as F


class TransformerModel(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 activation="relu",
                 tren_num_layers=1,
                 tren_num_heads=1,
                 tren_dim_ffw=256,
                 tren_dropout_pe=0.1,
                 tren_dropout=0.1,
                 tren_layer_norm_eps=1e-5,
                 tren_norm_first=True,
                 tren_add_bias=True,
                 tren_add_tailnorm=True):
        super().__init__()

        # Feature extraction.
        self.linear = nn.Linear(in_channels, inter_channels)
        self.activation = encoder.select_reluwise_activation(activation)

        # Transformer-Encoder.
        enlayer = encoder.TransformerEncoderLayer(
                dim_model=inter_channels,
                num_heads=tren_num_heads,
                dim_ffw=tren_dim_ffw,
                dropout=tren_dropout,
                activation=activation,
                layer_norm_eps=tren_layer_norm_eps,
                norm_first=tren_norm_first,
                add_bias=tren_add_bias)
        self.tr_encoder = encoder.TransformerEncoder(
            encoder_layer=enlayer,
            num_layers=tren_num_layers,
            dim_model=inter_channels,
            dropout_pe=tren_dropout_pe,
            layer_norm_eps=tren_layer_norm_eps,
            norm_first=tren_norm_first,
            add_bias=tren_add_bias,
            add_tailnorm=tren_add_tailnorm)
        
        head_settings = GPoolRecognitionHeadSettings(
            in_channels=inter_channels,
            out_channels=out_channels
        )

        self.head = GPoolRecognitionHead(head_settings)

    def forward(self,
                feature,
                feature_causal_mask=None,
                feature_pad_mask=None):
        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape(N, T, -1)

        feature = self.linear(feature)
        if torch.isnan(feature).any():
            raise ValueError()
        feature = self.activation(feature)
        if torch.isnan(feature).any():
            raise ValueError()

        feature = self.tr_encoder(
            feature=feature,
            causal_mask=feature_causal_mask,
            src_key_padding_mask=feature_pad_mask)
        if torch.isnan(feature).any():
            raise ValueError()

        # `[N, T, C] -> [N, C, T]`
        logit = self.head(feature.permute([0, 2, 1]), feature_pad_mask)
        if torch.isnan(feature).any():
            raise ValueError()
        return logit
    



class ConfiguredModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class GPoolRecognitionHeadSettings(ConfiguredModel):
    in_channels: int = 64
    out_channels: int = 64

    def build_layer(self):
        return GPoolRecognitionHead(self)

class GPoolRecognitionHead(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, GPoolRecognitionHeadSettings)
        self.settings = settings

        self.head = nn.Linear(settings.in_channels, settings.out_channels)
        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.head.weight,
                        mean=0.,
                        std=math.sqrt(1. / self.settings.out_channels))

    def forward(self, feature, feature_pad_mask=None):
        # Averaging over temporal axis.
        # `[N, C, T] -> [N, C, 1] -> [N, C]`
        if feature_pad_mask is not None:
            tlength = feature_pad_mask.sum(dim=-1)
            feature = feature * feature_pad_mask.unsqueeze(1)
            feature = feature.sum(dim=-1) / tlength.unsqueeze(-1)
        else:
            feature = F.avg_pool1d(feature, kernel_size=feature.shape[-1])
        feature = feature.reshape(feature.shape[0], -1)

        # Predict.
        feature = self.head(feature)
        return feature