import one_dcnn_transformer_encoder.models.one_dcnn as cnn
import cnn_transformer.models.transformer_encoer as encoder
from torch import nn
import torch

class OnedCNNTransformerEncoderModel(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        inter_channels,
        stride,
        padding,
        activation="relu",
        tren_num_layers=1,
        tren_num_heads=1,
        tren_dim_ffw=256,
        tren_dropout=0.1,
        tren_norm_eps=1e-5,
        batch_first= True,
        tren_norm_first=True,
        tren_add_bias=True,
        num_classes=100,
        blank_idx=1,
    ):
        super().__init__()

        #1DCNNモデル
        self.cnn_model = cnn.resnet18_1d(num_classes=num_classes, in_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        print(self.cnn_model)


        #TransformerEncoderモデル
        enlayer = nn.TransformerEncoderLayer(
            d_model=inter_channels,
            nhead=tren_num_heads,
            dim_feedforward=tren_dim_ffw,
            dropout=tren_dropout,
            activation=activation,
            layer_norm_eps=tren_norm_eps,
            batch_first=batch_first,
            norm_first=tren_norm_first,
            bias=tren_add_bias
        )
        self.tr_encoder = nn.TransformerEncoder(
            encoder_layer=enlayer,
            num_layers=tren_num_layers
        )
        
         # クラス分類用の線形層
        self.classifier = nn.Linear(inter_channels, num_classes)

        # ログソフトマックス（CTC損失用）
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # CTC損失関数
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)


    def forward(self,
               src_feature,
               tgt_feature,
               src_causal_mask,
               src_padding_mask,
               input_lengths,
               target_lengths,
               is_training):
        """
        src_feature:[batch, C, T, J]
        tgt_feature:[batch, max_len]答えをバッチサイズの最大値に合わせてpaddingしている
        input_lengths:[batch]1DCNNに通した後のデータ
        target_lengths:[batch]tgt_featureで最大値に伸ばしたので本当の長さ
        """
        N, C, T, J = src_feature.shape
        src_feature = src_feature.permute(0, 3, 1, 2).contiguous().view(N, C * J, T)
        cnn_out = self.cnn_model(src_feature) #[batch, 512, T']

        cnn_out = cnn_out.permute(0, 2, 1)     # (batch, T', 512)

        tr_out = self.tr_encoder(
            src=cnn_out,
            mask=src_causal_mask,
            src_key_padding_mask=src_padding_mask
        )  # (batch, T', inter_channels)

        logits = self.classifier(tr_out)  # (batch, T', num_classes)

        # ログソフトマックスを適用
        log_probs = self.log_softmax(logits)  # (batch, T', num_classes)

        # CTC損失の入力形式に変換: (T', batch, C)
        log_probs = log_probs.permute(1, 0, 2)  # (T', batch, num_classes)

        if is_training:
            # CTC損失を計算
            loss = self.ctc_loss(
                log_probs,         # (T', batch, C)
                tgt_feature,           # (batch, max_target_length)
                input_lengths,     # (batch,)
                target_lengths     # (batch,)
            )
            return loss, log_probs
        else:
            return log_probs


# # ダミーデータの生成
# def generate_dummy_data(batch_size=8, in_channels=3, T=100, J=3, num_classes=100, max_target_length=20):
#     """
#     ダミーデータを生成します。
#     Args:
#         batch_size: バッチサイズ
#         in_channels: 入力チャンネル数
#         T: 元の系列長
#         J: ジョイント数（または他の次元）
#         num_classes: クラス数
#         max_target_length: ターゲット系列の最大長
#     Returns:
#         src_feature: [batch, C, T, J]
#         tgt_feature: [batch, max_target_length]
#         input_lengths: [batch]
#         target_lengths: [batch]
#     """
#     # 入力特徴量の生成
#     src_feature = torch.randn(batch_size, in_channels, T, J)

#     # ターゲットラベルの生成（0からnum_classes-1の整数）
#     tgt_feature = torch.randint(low=0, high=num_classes, size=(batch_size, max_target_length), dtype=torch.long)

#     # 入力系列の長さ（CNNを通った後の長さを仮定）
#     # 仮にCNNがストライド1で系列長を維持すると仮定
#     input_lengths = torch.full((batch_size,), T//2, dtype=torch.long)

#     # ターゲット系列の長さをランダムに生成（5からmax_target_length）
#     target_lengths = torch.randint(low=5, high=max_target_length + 1, size=(batch_size,), dtype=torch.long)
#     print(tgt_feature)

#     return src_feature, tgt_feature, input_lengths, target_lengths

# # モデルの初期化
# def initialize_model():
#     model = OnedCNNTransformerEncoderModel(
#         in_channels=9,
#         kernel_size=3,
#         inter_channels=512,      # Transformerの入力次元
#         stride=1,
#         padding=1,
#         activation="relu",
#         tren_num_layers=2,       # 例として2層のTransformer
#         tren_num_heads=8,        # マルチヘッドの数
#         tren_dim_ffw=2048,       # フィードフォワードの次元
#         tren_dropout=0.1,
#         tren_norm_eps=1e-5,
#         batch_first=True,
#         tren_norm_first=True,
#         tren_add_bias=True,
#         num_classes=100,
#         blank_idx=1,
#     )
#     return model

# # テストの実行
# def test_model():
#     # デバイスの設定
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # ダミーデータの生成
#     src_feature, tgt_feature, input_lengths, target_lengths = generate_dummy_data()

#     # モデルの初期化
#     model = initialize_model().to(device)

#     # データをデバイスに移動
#     src_feature = src_feature.to(device)
#     tgt_feature = tgt_feature.to(device)
#     input_lengths = input_lengths.to(device)
#     target_lengths = target_lengths.to(device)

#     # モデルを訓練モードに設定
#     model.train()

#     # オプティマイザの設定
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#     # フォワードパスの実行
#     print(src_feature.shape)
#     loss, log_probs = model.forward(
#         src_feature=src_feature,
#         tgt_feature=tgt_feature,
#         src_causal_mask=None,      # オプション、今回は使用しない
#         src_padding_mask=None,     # オプション、今回は使用しない
#         input_lengths=input_lengths,
#         target_lengths=target_lengths
#     )

#     print(f"Initial loss: {loss.item()}")

#     # バックプロパゲーションとオプティマイザのステップ
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print("Forward pass and optimization step completed.")

# if __name__ == "__main__":
#     test_model()