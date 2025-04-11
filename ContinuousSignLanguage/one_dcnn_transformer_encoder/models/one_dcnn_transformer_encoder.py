import one_dcnn_transformer_encoder.models.one_dcnn as cnn
import cnn_transformer.models.transformer_encoer as encoder
from torch import nn
import torch


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

        # ログソフトマックス（CTC損失用）
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # CTC損失関数
        print(f"blank_idx: {blank_idx}")
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=False)

    def forward(
        self,
        src_feature,
        tgt_feature,
        src_causal_mask,
        src_padding_mask,
        input_lengths,
        target_lengths,
        mode,
        blank_id=100,
    ):
        """
        src_feature:[batch, C, T, J]
        tgt_feature:[batch, max_len]答えをバッチサイズの最大値に合わせてpaddingしている
        input_lengths:[batch]1D CNNに通した後のデータ
        target_lengths:[batch]tgt_featureで最大値に伸ばしたので本当の長さ
        """
        N, C, T, J = src_feature.shape
        src_feature = src_feature.permute(0, 3, 1, 2).contiguous().view(N, C * J, T)
        # 入力データにNaN値があるかチェック
        if torch.isnan(src_feature).any():
            print("警告: 入力src_featureにNaN値が検出されました")

        # 入力データの統計情報を表示

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
        src_feature = torch.clamp(src_feature, min=-10.0, max=10.0)
        # src_featureを転置して(batch, T, C*J)の形状にする

        cnn_out = self.cnn_model(src_feature)  # [batch, 512, T']
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
                # シーケンスが長くなった場合はパディング（通常はありえないが念のため）
                else:
                    padding = torch.zeros(
                        src_padding_mask.shape[0],
                        cnn_seq_len - original_seq_len,
                        device=src_padding_mask.device,
                        dtype=src_padding_mask.dtype,
                    )
                    src_padding_mask = torch.cat([src_padding_mask, padding], dim=1)

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

        logits = self.classifier(tr_out)  # (batch, T', num_classes)

        # ログソフトマックスを適用
        log_probs = self.log_softmax(logits)  # (batch, T', num_classes)

        # CTC損失の入力形式に変換: (T', batch, C)
        log_probs = log_probs.permute(1, 0, 2)  # (T', batch, num_classes)


        if mode == "train":
            # CTC損失を計算
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
            decoded_sequences = beam_search_decode(
                log_probs, beam_width=10, blank_id=blank_id
            )
            return loss, decoded_sequences
        elif mode == "test":
            decoded_sequences = beam_search_decode(
                log_probs, beam_width=10, blank_id=blank_id
            )
            return decoded_sequences


def beam_search_decode(log_probs, beam_width=10, blank_id=0):
    """
    ビームサーチを使用したCTCデコーディング

    Args:
        log_probs (torch.Tensor): ログ確率テンソル (T', batch, num_classes)
        beam_width (int): ビームの幅
        blank_id (int): ブランクラベルのID

    Returns:
        List[List[int]]: デコードされたシーケンス
    """
    batch_size = log_probs.size(1)
    decoded_sequences = []

    for b in range(batch_size):
        # 1バッチごとにビームサーチを実行
        seq_probs = log_probs[:, b, :]

        # ビーム初期化
        beams = [([], 0.0)]

        for t in range(seq_probs.size(0)):
            new_beams = []

            for prefix, score in beams:
                # 現在のタイムステップの確率を取得
                probs = seq_probs[t]

                # 各クラスについて新しいビームを生成
                for c in range(len(probs)):
                    # 新しいスコアを計算
                    new_score = score + probs[c].item()

                    # 同じラベルの連続を避ける
                    if c == blank_id:
                        new_beam = (prefix, new_score)
                    elif not prefix or c != prefix[-1]:
                        new_beam = (prefix + [c], new_score)
                    else:
                        continue

                    new_beams.append(new_beam)

            # ビームの幅を制限
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        # 最高スコアのビームを選択
        best_beam = max(beams, key=lambda x: x[1])[0]
        decoded_sequences.append(best_beam)
    return decoded_sequences


def soft_beam_search_decode(log_probs, beam_width=10, blank_id=0, lm_weight=0.5):
    """
    ソフトマックスと言語モデルを考慮したビームサーチ

    Args:
        log_probs (torch.Tensor): ログ確率テンソル
        beam_width (int): ビームの幅
        blank_id (int): ブランクラベルのID
        lm_weight (float): 言語モデルの重み

    Returns:
        List[List[int]]: デコードされたシーケンス
    """

    # 言語モデルの仮想的な実装（実際の言語モデルに置き換える）
    def language_model_score(sequence):
        # 単純な長さペナルティ（実際のタスクに合わせて調整）
        return -len(sequence)

    batch_size = log_probs.size(1)
    decoded_sequences = []

    for b in range(batch_size):
        # バッチごとの処理
        seq_probs = log_probs[:, b, :]

        # マルチビームサーチ
        beams = [([], 0.0)]

        for t in range(seq_probs.size(0)):
            candidates = []

            for prefix, score in beams:
                for c in range(seq_probs.size(1)):
                    # 新しいスコア計算
                    new_score = (
                        score
                        + seq_probs[t, c].item()  # 音響モデルスコア
                        + lm_weight
                        * language_model_score(prefix + [c])  # 言語モデルスコア
                    )

                    # ビーム候補に追加
                    candidates.append((prefix + [c], new_score))

            # ビームの幅で絞り込み
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

        # 最高スコアのビームを選択
        best_beam = max(beams, key=lambda x: x[1])[0]
        decoded_sequences.append(best_beam)

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
