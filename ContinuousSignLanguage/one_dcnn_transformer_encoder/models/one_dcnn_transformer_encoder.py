import one_dcnn_transformer_encoder.models.one_dcnn as cnn
import cnn_transformer.models.transformer_encoer as encoder
from torch import nn
import torch
import one_dcnn_transformer_encoder.models.beam_search as beam_search


class OnedCNNTransformerEncoderModel(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        cnn_out_channels,
        stride,
        padding,
        dropout_rate,
        bias,
        resNet,
        activation="relu",
        tren_num_layers=1,
        tren_num_heads=1,
        tren_dim_ffw=256,
        tren_dropout=0.1,
        tren_norm_eps=1e-5,
        batch_first=True,
        tren_norm_first=True,
        tren_add_bias=True,
        num_classes=100,
        blank_idx=1,
    ):
        super().__init__()

        # blank_idxをクラス変数として保存
        self.blank_id = blank_idx

        # 1DCNNモデル
        # self.cnn_model = cnn.resnet18_1d(num_classes=num_classes, in_channels=in_channels, out_channels=cnn_out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dropout_rate=dropout_rate, bias=bias)
        self.cnn_model = cnn.create_simple_cnn1layer(
            in_channels=in_channels,
            out_channels=cnn_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dropout_rate=dropout_rate,
            bias=bias,
        )
        # self.cnn_model = cnn.create_simple_cnn2layer(in_channels=in_channels/2,mid_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dropout_rate=dropout_rate, bias=bias)

        inter_channels = cnn_out_channels if resNet == 0 else 512
        # TransformerEncoderモデル

        self.input_projection = nn.Linear(in_channels, inter_channels)

        enlayer = nn.TransformerEncoderLayer(
            d_model=inter_channels,
            nhead=tren_num_heads,
            dim_feedforward=tren_dim_ffw,
            dropout=tren_dropout,
            activation=activation,
            layer_norm_eps=tren_norm_eps,
            batch_first=batch_first,
            norm_first=tren_norm_first,
            bias=tren_add_bias,
        )
        self.tr_encoder = nn.TransformerEncoder(
            encoder_layer=enlayer, num_layers=tren_num_layers
        )
        
        # クラス分類用の線形層
        self.classifier = nn.Linear(inter_channels, num_classes)
        
        # クラス毎の相互抑制を減らすための層
        self.class_attention = nn.MultiheadAttention(embed_dim=num_classes, num_heads=1, batch_first=True)

        # 重みの初期化を改善
        self._initialize_weights()

        # ログソフトマックス（CTC損失用）
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        # 温度パラメータ（訓練可能）
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # CTC損失関数 - 不均衡なクラス分布に対処するためのオプション
        print(f"blank_idx: {blank_idx}")
        # 標準の損失
        self.ctc_loss = nn.CTCLoss(
            blank=blank_idx, zero_infinity=True, reduction="mean"
        )

        # トレーニング情報追跡
        self.train_steps = 0
        self.min_loss = float("inf")

    def _initialize_weights(self):
        """
        重みの初期化を改良した関数
        """
        # Classifier層の初期化 - より均一な分布を目指す
        # 各クラスが等しいチャンスを持つように初期化
        nn.init.xavier_uniform_(
            self.classifier.weight, gain=0.01
        )  # gainを小さくして初期値を抑制
        if self.classifier.bias is not None:
            # バイアスは最初はゼロに近い値に設定
            nn.init.constant_(self.classifier.bias, 0)

            # ブランク以外のクラスにわずかに正のバイアスを与える
            with torch.no_grad():
                for i in range(self.classifier.bias.size(0)):
                    if i != self.blank_id:  # ブランク以外のクラス
                        self.classifier.bias[i] += 0.1  # 小さな正のバイアス

        print("モデルの重み初期化を実行しました（改良版）")

        # 改良された重みの統計情報を表示
        with torch.no_grad():
            classifier_weights = self.classifier.weight.data
            print(
                f"分類層の重み統計: 平均={classifier_weights.mean().item():.4f}, 標準偏差={classifier_weights.std().item():.4f}"
            )

            # クラス別のバイアス値を確認（もしバイアスがある場合）
            if self.classifier.bias is not None:
                blank_bias = self.classifier.bias[self.blank_id].item()
                other_bias_mean = torch.mean(
                    torch.cat(
                        [
                            self.classifier.bias[: self.blank_id],
                            self.classifier.bias[self.blank_id + 1 :],
                        ]
                    )
                ).item()

                print(f"ブランクラベル({self.blank_id})のバイアス: {blank_bias:.4f}")
                print(f"他のクラスの平均バイアス: {other_bias_mean:.4f}")

    def forward(
        self,
        src_feature,
        tgt_feature,
        src_causal_mask,
        src_padding_mask,
        input_lengths,
        target_lengths,
        mode,
        blank_id=None,
        current_epoch=None,
    ):
        """
        src_feature:[batch, C, T, J]
        tgt_feature:[batch, max_len]答えをバッチサイズの最大値に合わせてpaddingしている
        input_lengths:[batch]1D CNNに通した後のデータ
        target_lengths:[batch]tgt_featureで最大値に伸ばしたので本当の長さ
        current_epoch: 現在のエポック番号（グラフ作成時に使用）
        """
        # blank_idが指定されていない場合はクラス変数を使用
        if blank_id is None:
            blank_id = self.blank_id

        # シーケンス長とバッチサイズを記録（デバッグ用）
        print(
            f"CNNに入力される長さの範囲: {input_lengths.min().item()} - {input_lengths.max().item()}"
        )
        print(
            f"ターゲット長の範囲: {target_lengths.min().item()} - {target_lengths.max().item()}"
        )

        N, C, T, J = src_feature.shape
        src_feature = src_feature.permute(0, 3, 1, 2).contiguous().view(N, C * J, T)
        # 入力データにNaN値があるかチェック
        if torch.isnan(src_feature).any():
            print("警告: 入力src_featureにNaN値が検出されました")

        # 無限大の値があるかチェック
        if torch.isinf(src_feature).any():
            print("警告: 入力src_featureに無限大の値が検出されました")
        # 数値安定性のための正規化を追加
        # src_feature形状は[batch, C*J, T]なので、dim=2ではなくdim=1に沿って正規化
        eps = 1e-5
        src_mean = src_feature.mean(dim=1, keepdim=True)
        src_std = src_feature.std(dim=1, keepdim=True) + eps
        src_feature = (src_feature - src_mean) / src_std

        # 大きすぎる値や小さすぎる値をクリップ
        src_feature = torch.clamp(src_feature, min=-5.0, max=5.0)
        print(src_feature.shape, "src_feature.shape")
        cnn_out = self.cnn_model(src_feature)  # return [batch, 512, T']
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, T', 512)
        if torch.isnan(cnn_out).any():
            print("層1の出力にNaN値が検出されました")
            # 問題のある入力値の確認
            problem_indices = torch.where(torch.isnan(cnn_out))
            print(
                f"問題がある入力の要素: {cnn_out[problem_indices[0][0], problem_indices[1][0], problem_indices[2][0]]}"
            )

        # パディングマスクのサイズをCNN出力のシーケンス長に合わせる
        if src_padding_mask is not None:
            # 元のシーケンス長
            original_seq_len = src_padding_mask.shape[1]
            # CNNを通過した後のシーケンス長
            cnn_seq_len = cnn_out.shape[1]
            if original_seq_len != cnn_seq_len:
                print(
                    f"パディングマスクのサイズを調整: {original_seq_len} -> {cnn_seq_len}"
                )
                # シーケンスが短くなった場合は切り詰め
                if original_seq_len > cnn_seq_len:
                    src_padding_mask = src_padding_mask[:, :cnn_seq_len]

                else: # シーケンスが長くなった場合はパディング
                    padding_length = cnn_seq_len - original_seq_len
                    padding = torch.zeros(src_padding_mask.size(0), padding_length, dtype=src_padding_mask.dtype, device=src_padding_mask.device)
                    src_padding_mask = torch.cat((src_padding_mask, padding), dim=1)
 
        tr_out = self.tr_encoder(
            src=cnn_out, mask=src_causal_mask, src_key_padding_mask=src_padding_mask
        )  # (batch, T', inter_channels)

        if torch.isnan(tr_out).any():
            print("Transformerの出力にNaN値が検出されました")
            # 問題のある入力値の確認
            problem_indices = torch.where(torch.isnan(tr_out))
            print(
                f"問題がある入力の要素: {tr_out[problem_indices[0][0], problem_indices[1][0], problem_indices[2][0]]}"
            )

        # クラス分類
        logits = self.classifier(tr_out)  # (batch, T', num_classes)
        
        # マルチヘッドアテンション機構を使って、クラス間の関係性を学習
        # 各時間フレームでの予測を改善し、複数クラスの検出を促進
        # logits: (batch, T', num_classes)
        attended_logits, _ = self.class_attention(logits, logits, logits)
        
        # 元のロジットとアテンション後のロジットを組み合わせる
        # これにより、クラス間の排他的な競合を減らし、複数の手話単語の検出を可能に
        final_logits = logits + 0.5 * attended_logits  # 重み付け係数は調整可能

        # ログソフトマックスを適用する前に、出力分布のチェック
        with torch.no_grad():
            # 訓練前の出力分布を確認
            max_logit = torch.max(final_logits).item()
            min_logit = torch.min(final_logits).item()
            mean_logit = torch.mean(final_logits).item()
            std_logit = torch.std(final_logits).item()
            print(
                f"ロジット統計: 最大={max_logit:.4f}, 最小={min_logit:.4f}, 平均={mean_logit:.4f}, 標準偏差={std_logit:.4f}"
            )

            # クラスごとの平均値を確認
            class_means = torch.mean(final_logits, dim=(0, 1))
            top5_means, top5_indices = torch.topk(class_means, 5)
            print(f"平均値が高い上位5クラス: {top5_indices.tolist()}")
            print(f"上位5クラスの平均値: {top5_means.tolist()}")

            # 標準偏差が小さすぎる場合は警告
            if std_logit < 0.1:
                print(
                    "警告: ロジットの標準偏差が非常に小さいです。モデルが学習できていない可能性があります。"
                )

        # 出力が偏りすぎないように動的な温度パラメータでスケーリング
        # 訓練モードでは学習可能な温度パラメータを使用
        if mode == "train":
            # 現在のエポックに基づいて温度を調整（早期のエポックでは高温に）
            if current_epoch is not None and current_epoch < 5:
                # 早期のエポックでは分布をより均一にして、多様な学習を促進
                temp = torch.clamp(self.temperature * 1.5, min=1.0, max=2.0)
            else:
                temp = self.temperature
            scaled_logits = final_logits / temp
        else:
            # 評価・テストモードでは固定の温度で、複数クラスの検出を促進
            temp = 1.5  # >1.0で分布がより均一に、<1.0で最大値がより強調される
            scaled_logits = final_logits / temp
            print(f"温度パラメータ {temp} でロジットをスケーリングしました")

        # ログソフトマックスを適用
        log_probs = self.log_softmax(scaled_logits)  # (batch, T', num_classes)

        # CTC損失の入力形式に変換: (T', batch, C)
        log_probs = log_probs.permute(1, 0, 2)  # (T', batch, num_classes)

        if mode == "train":
            # CTC損失を計算
            print(tgt_feature, "tgt_feature")
            loss = self.ctc_loss(
                log_probs,  # (T', batch, C)
                tgt_feature,  # (batch, max_target_length)
                input_lengths,  # (batch,)
                target_lengths,  # (batch,)
            )
            return loss, log_probs
        elif mode == "eval":
            loss = self.ctc_loss(
                log_probs,  # (T', batch, C)
                tgt_feature,  # (batch, max_target_length)
                input_lengths,  # (batch,)
                target_lengths,  # (batch,)
            )
            # エポック情報をbeam_searchに渡す
            decoded_sequences = beam_search.beam_search_decode(
                log_probs,
                beam_width=10,
                blank_id=self.blank_id,
                current_epoch=current_epoch,
            )
            return loss, decoded_sequences
        elif mode == "test":
            # エポック情報をbeam_searchに渡す
            decoded_sequences = beam_search.beam_search_decode(
                log_probs,
                beam_width=10,
                blank_id=self.blank_id,
                current_epoch=current_epoch,
            )
            return decoded_sequences


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
