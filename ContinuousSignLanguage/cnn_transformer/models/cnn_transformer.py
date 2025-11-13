import cnn_transformer.models.transformer_modules as modules
import cnn_transformer.models.cnn_model as cnn_model
import cnn_transformer.models.transformer_encoer as encoder
import cnn_transformer.models.transformer_decoder as decoder
import cnn_transformer.continuous_sign_language_cnn_transformer.config as model_config
import cnn_transformer.models.spatial_feature_function as spatial_feature_function
from torch import nn
import torch
import numpy as np

class CNNTransformerModel(nn.Module):
    def __init__(
            self,
            in_channels,
            kernel_size,
            inter_channels,
            stride,
            out_channels,
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
            trde_add_tailnorm=True):
        super().__init__()
        
        #1DCNN block
        cnn = cnn_model.CNNFeatureExtractor(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=kernel_size,
            stride=stride
        )
        self.cnn_extractor = cnn_model.CNNLocalFeatureExtractor(
            cnn=cnn
        )

        self.linear = nn.Linear(inter_channels, inter_channels)
        self.activation = modules.select_reluwise_activation(activation)

        # Transformer-Encoder.
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

        # Transformer-Decoder.
        delayer = decoder.TransformerDecoderLayer(
            dim_model=inter_channels,
            num_heads=trde_num_heads,
            dim_ffw=trde_dim_ffw,
            dropout=trde_dropout,
            activation=activation,
            norm_type_sattn=trde_norm_type_sattn,
            norm_type_cattn=trde_norm_type_cattn,
            norm_type_ffw=trde_norm_type_ffw,
            norm_eps=trde_norm_eps,
            norm_first=trde_norm_first,
            add_bias=trde_add_bias)
        self.tr_decoder = decoder.TransformerDecoder(
            decoder_layer=delayer,
            out_channels=out_channels,
            num_layers=trde_num_layers,
            dim_model=inter_channels,
            dropout_pe=trde_dropout_pe,
            norm_type_tail=trde_norm_type_tail,
            norm_eps=trde_norm_eps,
            norm_first=trde_norm_first,
            add_bias=trde_add_bias,
            add_tailnorm=trde_add_tailnorm,
            padding_val=padding_val)
        
    def forward(self,
                src_feature,
                tgt_feature,
                src_causal_mask,
                src_padding_mask,
                tgt_causal_mask,
                tgt_padding_mask):
        """
        フォワードパスの実装。
        
        Args:
            src_feature (Tensor): ソースの特徴 [N, C, T, J]
            tgt_feature (Tensor, optional): ターゲットの特徴 [N, C, T, J]
            src_causal_mask (Tensor, optional): ソースへの因果マスク
            src_padding_mask (Tensor, optional): ソースのパディングマスク
            tgt_causal_mask (Tensor, optional): ターゲットへの因果マスク
            tgt_padding_mask (Tensor, optional): ターゲットのパディングマスク
    """
        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = src_feature.shape #バッチサイズ, チャネル数, フレーム, 骨格座標
        # 形状を [N, C, T, J] -> [N, C* J, T] に変換
        spatial_feature = spatial_feature_function.get_spatial(src_feature)
        spatial_feature = spatial_feature.permute(0, 2, 1)
        src_feature = src_feature.view(N, C * J, T)
        #CNNの処理 [N, L, T] -> [N, T', L']
        src_feature = torch.cat((src_feature, spatial_feature), dim=1)
        src_feature = self.cnn_extractor(src_feature)
        print(src_feature.shape)
        enc_feature = self.tr_encoder(
            feature=src_feature,
            causal_mask=src_causal_mask,
            src_key_padding_mask=src_padding_mask)

        if torch.isnan(enc_feature).any():
            print("Transformerの出力にNaN値が検出されました")
            # 問題のある入力値の確認
            problem_indices = torch.where(torch.isnan(enc_feature))
            print(enc_feature)
            print(f"問題がある入力の要素: {enc_feature[problem_indices[0][0], problem_indices[1][0], problem_indices[2][0]]}")
            exit(0)

        preds = self.tr_decoder(tgt_feature=tgt_feature,
                                enc_feature=enc_feature,
                                tgt_causal_mask=tgt_causal_mask,
                                enc_tgt_causal_mask=None,
                                tgt_key_padding_mask=tgt_padding_mask,
                                enc_key_padding_mask=src_padding_mask)
        # `[N, T, C]`
        return preds

    def inference(self,
                  src_feature,
                  start_id,
                  end_id,
                  src_padding_mask=None,
                  max_seqlen=62):
        """Forward computation for test.
        """

        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = src_feature.shape #バッチサイズ, チャネル数, フレーム, 骨格座標
        # 形状を [N, C, T, J] -> [N, C* J, T] に変換
        spatial_feature = spatial_feature_function.get_spatial(src_feature)
        spatial_feature = spatial_feature.permute(0, 2, 1)
        src_feature = src_feature.view(N, C * J, T)
        src_feature = torch.cat((src_feature, spatial_feature), dim=1)

        #CNNの処理 [N, L, T] -> [N, T', L']
        src_feature = self.cnn_extractor(src_feature)
        enc_feature = self.tr_encoder(
            feature=src_feature,
            causal_mask=None,
            src_key_padding_mask=src_padding_mask)

        # Apply decoder.
        dec_inputs = torch.tensor([start_id]).to(src_feature.device)
        # `[N, T]`
        dec_inputs = dec_inputs.reshape([1, 1])
        preds = None
        pred_ids = [start_id]
        for _ in range(max_seqlen):
            pred = self.tr_decoder(
                tgt_feature=dec_inputs,
                enc_feature=enc_feature,
                tgt_causal_mask=None,
                enc_tgt_causal_mask=None,
                tgt_key_padding_mask=None,
                enc_key_padding_mask=src_padding_mask)
            # Extract last prediction.
            pred = pred[:, -1:, :]
            # `[N, T, C]`
            if preds is None:
                preds = pred
            else:
                # Concatenate last elements.
                preds = torch.cat([preds, pred], dim=1)

            pid = torch.argmax(pred, dim=-1)
            dec_inputs = torch.cat([dec_inputs, pid], dim=-1)

            pid = pid.reshape([1]).detach().cpu().numpy()[0]
            pred_ids.append(int(pid))
            if int(pid) == end_id:
                break

        # `[N, T]`
        pred_ids = np.array([pred_ids])
        return pred_ids, preds



