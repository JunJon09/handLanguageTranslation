import CNN_BiLSTM.models.one_dcnn as cnn
import CNN_BiLSTM.models.BiLSTM as BiLSTM
import CNN_BiLSTM.models.TransformerLayer as TransformerLayer
import CNN_BiLSTM.models.decode as decode
import CNN_BiLSTM.models.criterions as criterions
import logging

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
        self.scale = hidden_dim**-0.5
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x, return_attention=False):
        # x: (T, B, D) → (B, T, D)
        x = x.transpose(0, 1)
        residual = x
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, T, T)
        attn_weights = F.softmax(attn, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = self.output_proj(out)
        out = self.layer_norm(residual + out)
        # (B, T, D) → (T, B, D)
        out = out.transpose(0, 1)

        if return_attention:
            return out, attn_weights  # attn_weights: (B, T, T)
        return out


class HierarchicalTemporalModule(nn.Module):
    """階層的時間モデリングを行うモジュール - 異なる時間スケールでの特徴抽出"""

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        # 局所的特徴抽出（小さいカーネル）
        self.local_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,  # 短い時間スケール
            padding=1,
            stride=1,
        )

        # 中間的特徴抽出（中程度のカーネル、拡張畳み込み）
        self.mid_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=5,  # 中程度の時間スケール
            padding=4,  # padding = (kernel_size-1) * dilation / 2
            stride=1,
            dilation=2,  # 拡張畳み込み - 受容野を広げる
        )

        # グローバル特徴抽出（大きいカーネル、大きい拡張率）
        self.global_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=7,  # 長い時間スケール
            padding=9,  # padding = (kernel_size-1) * dilation / 2
            stride=1,
            dilation=3,  # さらに広い受容野
        )

        # 特徴統合層
        self.fusion = nn.Linear(hidden_dim * 3, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        入力: x (T, B, D) - 時間優先の形状
        出力: (T', B, D) - 異なる時間スケールで処理され融合された特徴
        """
        # 入力形状を変換: (T, B, D) -> (B, D, T) [Conv1d用]
        batch_size = x.size(1)
        x_permuted = x.permute(1, 2, 0)  # (T, B, D) -> (B, D, T)

        # 異なる時間スケールでの特徴抽出
        local_feat = F.relu(self.local_conv(x_permuted))  # 局所的特徴
        mid_feat = F.relu(self.mid_conv(local_feat))  # 中間的特徴
        global_feat = F.relu(self.global_conv(mid_feat))  # グローバル特徴

        # 時間次元の長さを確認
        local_T = local_feat.size(2)
        mid_T = mid_feat.size(2)
        global_T = global_feat.size(2)

        # 時間次元を最小値に揃える
        min_T = min(local_T, mid_T, global_T)
        local_feat = local_feat[:, :, :min_T]
        mid_feat = mid_feat[:, :, :min_T]
        global_feat = global_feat[:, :, :min_T]

        # (B, D, T) -> (B, T, D)に変換
        local_feat = local_feat.permute(0, 2, 1)
        mid_feat = mid_feat.permute(0, 2, 1)
        global_feat = global_feat.permute(0, 2, 1)

        # 特徴結合と融合
        concat_feat = torch.cat(
            [local_feat, mid_feat, global_feat], dim=2
        )  # (B, T, 3*H)
        fused_feat = self.dropout(self.fusion(concat_feat))  # (B, T, D)

        # 残差接続のための元の入力を取得（時間次元を揃える）
        x_btd = x.permute(1, 0, 2)[:, :min_T, :]  # (B, T', D)

        # 残差接続とレイヤー正規化
        output = self.layer_norm(fused_feat + x_btd)

        # (B, T, D) -> (T, B, D)に戻す
        return output.permute(1, 0, 2)


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
        temporal_model_type="bilstm",  # "bilstm", "transformer", "multiscale_transformer"
    ):
        super().__init__()

        # blank_idxをクラス変数として保存
        self.blank_id = 0
        self.temporal_model_type = temporal_model_type

        # 1DCNNモデル
        self.cnn_model = cnn.DualCNNWithCTC(
            skeleton_input_size=in_channels,
            hand_feature_size=24,
            skeleton_hidden_size=cnn_out_channels//2,
            hand_hidden_size=cnn_out_channels//2,
            fusion_hidden_size=cnn_out_channels,
            num_classes=num_classes,
            blank_idx=blank_idx,
        ).to("cpu")

        # self.cnn_model = cnn.DualMultiScaleTemporalConv(
        #     skeleton_input_size=in_channels,
        #     spatial_input_size=24,
        #     skeleton_hidden_size=cnn_out_channels // 2,
        #     spatial_hidden_size=cnn_out_channels // 2,
        #     fusion_hidden_size=cnn_out_channels,
        #     blank_idx=blank_idx,
        # )

        self.loss = dict()
        self.criterion_init()
        # 損失の重みを調整 - Loss爆発とWERを改善するための設定
        self.loss_weights = {
            "ConvCTC": 0.8,  # CNNからの予測を重視してWER改善
            "SeqCTC": 1.2,  # BiLSTMからの予測をより重視
            "Dist": 0.15,  # 蒸留損失を大幅に削減してLoss爆発を防ぐ
        }

        # 時系列モデルの選択
        if temporal_model_type == "bilstm":
            self.temporal_model = BiLSTM.BiLSTMLayer(
                rnn_type="LSTM",
                input_size=cnn_out_channels,
                hidden_size=cnn_out_channels * 2,  # 隠れ層のサイズ
                num_layers=4,
                bidirectional=True,
            )
            temporal_output_size = cnn_out_channels * 2

        elif temporal_model_type == "transformer":
            self.temporal_model = TransformerLayer.TransformerLayer(
                input_size=cnn_out_channels,
                hidden_size=cnn_out_channels * 2,  # BiLSTMと同じ出力サイズ
                num_layers=tren_num_layers,
                num_heads=tren_num_heads,
                dropout=tren_dropout,
            )
            temporal_output_size = cnn_out_channels * 2

        elif temporal_model_type == "multiscale_transformer":
            self.temporal_model = TransformerLayer.MultiScaleTransformerLayer(
                input_size=cnn_out_channels,
                hidden_size=cnn_out_channels * 2,
                num_layers=tren_num_layers,
                num_heads=tren_num_heads,
                dropout=tren_dropout,
            )
            temporal_output_size = cnn_out_channels * 2

        else:
            raise ValueError(f"Unknown temporal_model_type: {temporal_model_type}")

        # 相関学習モジュールを追加
        self.spatial_correlation = SpatialCorrelationModule(input_dim=cnn_out_channels)

        # クラス分類用の線形層
        self.classifier = nn.Linear(temporal_output_size, num_classes)

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
        重みの初期化を強化した関数 - WER改善のためブランクトークンバイアスを大幅に改善
        """
        # Classifier層の初期化
        nn.init.xavier_uniform_(self.classifier.weight)

        # バイアスの特別な初期化 - ブランクを大幅に抑制し、他のクラスを強調
        if self.classifier.bias is not None:
            # ブランクトークン（インデックス0）のバイアスを大幅に下げる
            nn.init.constant_(self.classifier.bias, 0.1)  # 他のクラスを少し促進
            self.classifier.bias.data[0] = -2.0  # ブランクを強力に抑制（-1.0 → -2.0）

            logging.info(
                f"分類器バイアス初期化完了 - ブランク: {self.classifier.bias.data[0]:.3f}, 他: {self.classifier.bias.data[1]:.3f}"
            )

        # BiLSTM層の重み初期化
        if self.temporal_model_type == "bilstm":
            for name, param in self.temporal_model.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # forget gateのバイアスを1に設定（LSTM特有の初期化）
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)
        else:
            # Transformer層の重み初期化
            for module in self.temporal_model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.MultiheadAttention):
                    nn.init.xavier_uniform_(module.in_proj_weight)
                    nn.init.xavier_uniform_(module.out_proj.weight)

        # Spatial Correlation Moduleの初期化
        for module in [self.spatial_correlation]:
            for name, param in module.named_parameters():
                if "weight" in name and param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        logging.info(
            f"✅ {self.temporal_model_type.upper()}モデル重み初期化完了 - WER改善設定適用"
        )

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

        # cnn_out, cnn_logit, updated_lgt = self.cnn_model(
        #     skeleton_feat=src_feature,
        #     spatial_feature=spatial_feature,
        #     lgt=input_lengths,
        # )

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
        if hasattr(self, "_visualize_attention") and self._visualize_attention:
            cnn_out, attention_weights = self.spatial_correlation(
                cnn_out, return_attention=True
            )
            self.last_attention_weights = attention_weights  # 可視化用に保存
        else:
            cnn_out = self.spatial_correlation(cnn_out)

        # BiLSTM/Transformerの実行
        tm_outputs = self.temporal_model(cnn_out, updated_lgt)
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

            # トップ5で最も頻繁に出現するトークンを表示
            topk_values, topk_indices = torch.topk(probs.mean(dim=1), k=5, dim=1)

        # デコード実行
        pred = self.decoder.decode(outputs, updated_lgt, batch_first=False, probs=False)
        conv_pred = self.decoder.decode(
            cnn_logit, updated_lgt, batch_first=False, probs=False
        )
        logging.info(f"\n{self.temporal_model_type.upper()}後のデコード結果: {pred}")
        logging.info(f"CNN直後のデコード結果: {conv_pred}")

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
            # testモードでもlog_probsを返すオプションを追加
            return pred, conv_pred, outputs

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        conv_loss = 0
        seq_loss = 0
        dist_loss = 0
        for k, weight in self.loss_weights.items():
            if k == "ConvCTC":
                conv_loss += (
                    weight
                    * self.loss["CTCLoss"](
                        ret_dict["conv_logits"].log_softmax(-1),
                        label.cpu().int(),
                        ret_dict["feat_len"].cpu().int(),
                        label_lgt.cpu().int(),
                    ).mean()
                )
            elif k == "SeqCTC":
                seq_loss += (
                    weight
                    * self.loss["CTCLoss"](
                        ret_dict["sequence_logits"].log_softmax(-1),
                        label.cpu().int(),
                        ret_dict["feat_len"].cpu().int(),
                        label_lgt.cpu().int(),
                    ).mean()
                )
            elif k == "Dist":
                dist_loss += weight * self.loss["distillation"](
                    ret_dict["conv_logits"],
                    ret_dict["sequence_logits"].detach(),
                    use_blank=False,
                )
        loss = conv_loss + seq_loss + dist_loss
        logging.info(
            f"損失: ConvCTC={conv_loss.item()}, SeqCTC={seq_loss.item()}, Distillation={dist_loss.item()}"
        )
        logging.info(f"総損失: {loss.item()}")
        return loss

    def criterion_init(self):
        self.loss["CTCLoss"] = torch.nn.CTCLoss(reduction="none", zero_infinity=False)
        # 蒸留温度を下げてLoss安定化とWER改善
        self.loss["distillation"] = criterions.SeqKD(T=3)  # T=8から3に下げる
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

        # 入力の最大長
        max_input_len = src_feature_T

        # 純粋に割合ベースで計算
        scale_ratio = actual_seq_len / max_input_len

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

    def enable_attention_visualization(self):
        """Attention重みの可視化を有効化"""
        self._visualize_attention = True
        self.last_attention_weights = None

    def disable_attention_visualization(self):
        """Attention重みの可視化を無効化"""
        self._visualize_attention = False
        self.last_attention_weights = None

    def get_attention_weights(self):
        """最後に計算されたAttention重みを取得"""
        if (
            hasattr(self, "last_attention_weights")
            and self.last_attention_weights is not None
        ):
            return self.last_attention_weights.detach().cpu().numpy()
        else:
            return None
