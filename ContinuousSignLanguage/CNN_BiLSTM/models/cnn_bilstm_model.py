import CNN_BiLSTM.models.one_dcnn as cnn
import CNN_BiLSTM.models.BiLSTM as BiLSTM
import CNN_BiLSTM.models.decode as decode
import CNN_BiLSTM.models.criterions as criterions

from torch import nn
import torch
import torch.nn.functional as F


class SpatialCorrelationModule(nn.Module):
    """空間的相関関係を学習する自己注意モジュール"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: (T, B, D) → (B, T, D)
        x = x.transpose(0, 1)
        residual = x
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, T, T)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = self.output_proj(out)
        out = self.layer_norm(residual + out)
        # (B, T, D) → (T, B, D)
        return out.transpose(0, 1)


class CNNBiLSTMModel(nn.Module):
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
        blank_idx=0,
    ):
        super().__init__()

        # blank_idxをクラス変数として保存
        self.blank_id = 0

        # 1DCNNモデル
        self.cnn_model = cnn.DualCNNWithCTC(
            skeleton_input_size=in_channels,
            hand_feature_size=50,
            skeleton_hidden_size=512,
            hand_hidden_size=512,
            fusion_hidden_size=1024,
            num_classes=num_classes,
            blank_idx=blank_idx,
        ).to("cpu")

        self.loss = dict()
        self.criterion_init()
        # 損失の重みを調整 - BiLSTMの重みを増やす
        self.loss_weights = {
            "ConvCTC": 0.7,  # CNNからの出力の重みを下げる
            "SeqCTC": 1.0,  # BiLSTM後の出力の重みを上げる
            "Dist": 0.1,  # 知識蒸留（Distillation）損失に対する重み
        }

        self.temporal_model = BiLSTM.BiLSTMLayer(
            rnn_type="LSTM",
            input_size=1024,
            hidden_size=1024,  # 隠れ層のサイズ
            num_layers=4,
            bidirectional=True,
        )

        # 相関学習モジュールを追加
        self.spatial_correlation = SpatialCorrelationModule(input_dim=1024)

        # クラス分類用の線形層
        self.classifier = nn.Linear(1024, num_classes)

        # 重みの初期化を行う - コメントアウトを解除
        self._initialize_weights()

        # ログソフトマックス（CTC損失用）
        gloss_dict = {
            "<blank>": [0],
            "i": [1],
            "you": [2],
            "he": [3],
            "teacher": [4],
            "friend": [5],
            "today": [6],
            "tomorrow": [7],
            "school": [8],
            "hospital": [9],
            "station": [10],
            "go": [11],
            "come": [12],
            "drink": [13],
            "like": [14],
            "dislike": [15],
            "busy": [16],
            "use": [17],
            "meet": [18],
            "where": [19],
            "question": [20],
            "pp": [21],
            "with": [22],
            "water": [23],
            "food": [24],
            "money": [25],
            "apple": [26],
            "banana": [27],
            "<pad>": [28],
        }
        # 言語モデルを使用するgreedyデコーダーを使用
        self.decoder = decode.Decode(
            gloss_dict=gloss_dict,
            num_classes=num_classes,
            search_mode="greedy",  # 言語モデル統合型のグリーディサーチを使用
            blank_id=0,
        )

    def _initialize_weights(self):
        """
        重みの初期化を強化した関数 - ブランクトークンへのバイアスを大幅に減らす
        """
        # Classifier層の初期化
        nn.init.xavier_uniform_(self.classifier.weight)

        # バイアスの特別な初期化 - ブランクを抑制し、他のクラスを強調
        if self.classifier.bias is not None:
            # 最初にすべてのバイアスを0.5に設定（非ブランクに有利にする）
            nn.init.constant_(self.classifier.bias, 0.5)

            # ブランクのバイアスを大きな負の値に設定（選択されにくくする）
            with torch.no_grad():
                self.classifier.bias[self.blank_id] = -3.0

        print("モデルの重み初期化を実行しました（強化版 - ブランク抑制）")

        # 初期化後の重みとバイアスの統計を表示
        with torch.no_grad():
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

                print(f"ブランクラベルのバイアス: {blank_bias:.4f}")
                print(f"他のクラスの平均バイアス: {other_bias_mean:.4f}")

    def forward(
        self,
        src_feature,
        spatial_feature,
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
        spatial_feature = spatial_feature.permute(0, 2, 1)
        # 入力データにNaN値があるかチェック
        if torch.isnan(src_feature).any():
            print("警告: 入力src_featureにNaN値が検出されました")
            exit(1)
        # 無限大の値があるかチェック
        if torch.isinf(src_feature).any():
            print("警告: 入力src_featureに無限大の値が検出されました")

        # CNNモデルの実行
        cnn_out, cnn_logit, updated_lgt = self.cnn_model(
            skeleton_feat=src_feature, hand_feat=spatial_feature, lgt=input_lengths
        )

        # 実際のCNN出力形状に基づいて長さを計算
        updated_lgt = self.calculate_updated_lengths(input_lengths, T, cnn_out.shape[0])

        if torch.isnan(cnn_out).any():
            print("層1の出力にNaN値が検出されました")
            # 問題のある入力値の確認
            problem_indices = torch.where(torch.isnan(cnn_out))
            print(
                f"問題がある入力の要素: {cnn_out[problem_indices[0][0], problem_indices[1][0], problem_indices[2][0]]}"
            )
            exit(1)

        # 相関学習モジュールの適用
        cnn_out = self.spatial_correlation(cnn_out)

        # BiLSTMの実行
        print(f"updated_lgt: {updated_lgt}")
        print(f"updated_lgtの形状: {updated_lgt.shape}")
        print(f"updated_lgtの値: {updated_lgt.tolist()}")
        tm_outputs = self.temporal_model(cnn_out, updated_lgt)
        print(tm_outputs["predictions"].shape, "tm_outputs['predictions'].shape")
        outputs = self.classifier(tm_outputs["predictions"])  # (batch, T', num_classes)

        # ブランクトークン抑制 - 非常に強い場合のみ
        if mode == "test" and blank_id is not None:
            # テスト時のみブランクを直接抑制
            with torch.no_grad():
                # ブランクの確率を下げる（ロジット値を小さくする）
                outputs[:, :, blank_id] -= 5.0  # 強制的に値を下げる
                cnn_logit[:, :, blank_id] -= 5.0

        # 予測確率分布の分析
        with torch.no_grad():
            # ソフトマックス適用後の確率分布を取得
            probs = F.softmax(outputs, dim=-1)

            # ブランクトークンの平均確率を計算
            blank_probs = probs[:, :, self.blank_id].mean().item()

            # nan防止のために丁寧に他のトークンの平均確率を計算
            other_probs = 0.0
            count = 0

            # ブランク以前のトークンがあれば計算
            if self.blank_id > 0:
                pre_blank_probs = probs[:, :, : self.blank_id].mean().item()
                other_probs += pre_blank_probs
                count += 1

            # ブランク以降のトークンがあれば計算
            if self.blank_id < probs.shape[2] - 1:
                post_blank_probs = probs[:, :, self.blank_id + 1 :].mean().item()
                other_probs += post_blank_probs
                count += 1

            # 平均を計算（0除算防止）
            if count > 0:
                other_probs /= count

            # 比率計算（0除算防止）
            ratio = blank_probs / other_probs if other_probs > 0 else float("inf")

            print(f"\n予測確率分析:")
            print(f"ブランクトークン(<blank>)の平均確率: {blank_probs:.4f}")
            print(f"他のトークンの平均確率: {other_probs:.4f}")
            print(f"ブランク/他の比率: {ratio:.4f}")

            # トップ5で最も頻繁に出現するトークンを表示
            topk_values, topk_indices = torch.topk(probs.mean(dim=1), k=5, dim=1)
            print(f"\nトップ5の予測トークン（バッチ0）:")
            for i in range(5):
                token_idx = topk_indices[0, i].item()
                token_name = (
                    self.decoder.i2g_dict[token_idx]
                    if token_idx in self.decoder.i2g_dict
                    else "不明"
                )
                print(f"  {token_name}: {topk_values[0, i].item():.4f}")

        # デコード実行
        pred = self.decoder.decode(outputs, updated_lgt, batch_first=False, probs=False)
        conv_pred = self.decoder.decode(
            cnn_logit, updated_lgt, batch_first=False, probs=False
        )
        print(f"\nBiLSTM後のデコード結果: {pred}")
        print(f"CNN直後のデコード結果: {conv_pred}")

        if mode != "test":
            ret_dict = {
                "feat_len": updated_lgt,
                "conv_logits": cnn_logit,
                "sequence_logits": outputs,
                "conv_sents": conv_pred,
                "recognized_sents": pred,
            }
            loss = self.criterion_calculation(ret_dict, tgt_feature, target_lengths)
            return loss, outputs
        else:
            return pred, conv_pred

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == "ConvCTC":
                loss += (
                    weight
                    * self.loss["CTCLoss"](
                        ret_dict["conv_logits"].log_softmax(-1),
                        label.cpu().int(),
                        ret_dict["feat_len"].cpu().int(),
                        label_lgt.cpu().int(),
                    ).mean()
                )
            elif k == "SeqCTC":
                loss += (
                    weight
                    * self.loss["CTCLoss"](
                        ret_dict["sequence_logits"].log_softmax(-1),
                        label.cpu().int(),
                        ret_dict["feat_len"].cpu().int(),
                        label_lgt.cpu().int(),
                    ).mean()
                )
            elif k == "Dist":
                loss += weight * self.loss["distillation"](
                    ret_dict["conv_logits"],
                    ret_dict["sequence_logits"].detach(),
                    use_blank=False,
                )
        return loss

    def criterion_init(self):
        self.loss["CTCLoss"] = torch.nn.CTCLoss(reduction="none", zero_infinity=False)
        self.loss["distillation"] = criterions.SeqKD(T=8)
        return self.loss

    def calculate_updated_lengths(
        self, input_lengths, src_feature_T, cnn_output_shape_T
    ):
        """
        CNNの出力形状に基づいて、入力長から更新された長さ(updated_lgt)を割合ベースで計算する関数

        Args:
            input_lengths (torch.Tensor): 元の入力シーケンス長 [batch_size]
            src_feature_T (int): 入力特徴量のシーケンスの長さ(Maxで全て揃えているの) T
            cnn_output_shape_T (int): CNNの出力のシーケンスの長さ T

        Returns:
            torch.Tensor: 更新された長さ情報 [batch_size]
        """
        # CNNの出力からシーケンス長次元を特定
        actual_seq_len = cnn_output_shape_T  # CNNの出力シーケンス長

        print(f"CNNの出力シーケンス長: {actual_seq_len}")

        # 入力の最大長
        max_input_len = src_feature_T

        # 純粋に割合ベースで計算
        scale_ratio = actual_seq_len / max_input_len
        print(
            f"スケール比率: {scale_ratio:.4f} (出力長 {actual_seq_len} / 最大入力長 {max_input_len})"
        )

        # すべてのサンプルに対して割合を適用
        updated_lengths = []
        for length in input_lengths:
            # 入力長の割合と同じ割合で出力長を計算
            proportion = length.item() / max_input_len  # 元の長さの最大に対する割合
            updated_length = max(
                1, int(actual_seq_len * proportion)
            )  # 出力長に同じ割合を適用
            updated_lengths.append(updated_length)

        # テンソルに変換して返す
        device = input_lengths.device
        return torch.tensor(updated_lengths, dtype=torch.long, device=device)
