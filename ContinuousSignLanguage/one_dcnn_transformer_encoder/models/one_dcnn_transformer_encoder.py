import one_dcnn_transformer_encoder.models.one_dcnn as cnn
import cnn_transformer.models.transformer_encoer as encoder
from torch import nn

class OnedCNNTransformerEncoderModel(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        inter_channels,
        stride,
        padding,
        padding_val,
        activation="relu",
        tren_num_layers=1,
        tren_num_heads=1,
        tren_dim_ffw=256,
        tren_dropout_pe=0.1,
        tren_dropout=0.1,
        tren_norm_type_sattn="layer",
        tren_norm_type_ffw="layer",
        tren_norm_type_tail="layer",
        tren_norm_eps=1e-5,
        tren_norm_first=True,
        tren_add_bias=True,
        tren_add_tailnorm=True,
        trde_num_layers=1,
        trde_num_heads=1,
        trde_dim_ffw=256,
        trde_dropout_pe=0.1,
        trde_dropout=0.1,
        trde_norm_type_sattn="layer",
        trde_norm_type_cattn="layer",
        trde_norm_type_ffw="layer",
        trde_norm_type_tail="layer",
        trde_norm_eps=1e-5,
        trde_norm_first=True,
        trde_add_bias=True,
        trde_add_tailnorm=True,
        num_classes=100
    ):
        super().__init__()

        #1DCNNモデル
        self.cnn_model = cnn.resnet18_1d(num_classes=num_classes, in_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        print(self.cnn_model)

        #TransformerEncoderモデル
        enlayer = encoder.TransformerEncoderLayer(
            dim_model=inter_channels,
            num_heads=tren_num_heads,
            dim_ffw=tren_dim_ffw,
            dropout=tren_dropout,
            activation=activation,
            norm_type_sattn=tren_norm_type_sattn,
            norm_type_ffw=tren_norm_type_ffw,
            norm_eps=tren_norm_eps,
            norm_first=tren_norm_first,
            add_bias=tren_add_bias)
        
        self.tr_encoder = encoder.TransformerEncoder(
            encoder_layer=enlayer,
            num_layers=tren_num_layers,
            dim_model=inter_channels,
            dropout_pe=tren_dropout_pe,
            norm_type_tail=tren_norm_type_tail,
            norm_eps=tren_norm_eps,
            norm_first=tren_norm_first,
            add_bias=tren_add_bias,
            add_tailnorm=tren_add_tailnorm)


    def foward(self,
               src_feature,
               tgt_feature,
               src_causal_mask,
               src_padding_mask,
               tgt_causal_mask,
               tgt_padding_mask):
        pass

