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


class Model(nn.Module):
    def __init__(
        self,
        vocabulary,
        in_channels,
        hand_size,
        cnn_out_channels,
        cnn_dropout_rate,
        conv_type,
        use_bn,
        kernel_sizes=[10, 15, 20, 25, 30],
        num_layers=1,
        num_heads=1,
        dropout=0.1,
        num_classes=100,
        blank_id=0,
        cnn_model_type="DualCNNWithCTC", # "DualCNNWithCTC" or "DualMultiScaleTemporalConv"
        temporal_model_type="bilstm", # "bilstm" or "transformer"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cnn_out_channels = cnn_out_channels
        self.blank_id = blank_id
        self.cnn_model_type = cnn_model_type
        self.temporal_model_type = temporal_model_type
        self.num_classes = num_classes


        if self.cnn_model_type == "DualCNNWithCTC":
            self.cnn_model = cnn.DualCNNWithCTC(
            skeleton_input_size=self.in_channels,
            hand_feature_size=hand_size,
            skeleton_hidden_size=self.cnn_out_channels//2,
            hand_hidden_size=self.cnn_out_channels//2,
            fusion_hidden_size=self.cnn_out_channels,
            dropout_rate=cnn_dropout_rate,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=self.num_classes,
            blank_id=self.blank_id,
            ).to("cpu")

        elif self.cnn_model_type == "DualMultiScaleTemporalConv":
            self.cnn_model = cnn.DualMultiScaleTemporalConv(
                skeleton_input_size=self.in_channels,
                spatial_input_size=hand_size,
                skeleton_hidden_size=self.cnn_out_channels // 2,
                spatial_hidden_size=self.cnn_out_channels // 2,
                fusion_hidden_size=self.cnn_out_channels,
                skeleton_kernel_sizes=kernel_sizes,
                spatial_kernel_sizes=kernel_sizes,
                dropout_rate=cnn_dropout_rate,
                num_classes=self.num_classes,
                blank_id=self.blank_id,
            ).to("cpu")
        else:
            raise ValueError(f"Unknown cnn_model_type: {cnn_model_type}")


        # 時系列モデルの選択
        if temporal_model_type == "bilstm":
            self.temporal_model = BiLSTM.BiLSTMLayer(
                rnn_type="LSTM",
                input_size=self.cnn_out_channels,
                hidden_size=self.cnn_out_channels * 2,  # 隠れ層のサイズ
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=True,
            )
            temporal_output_size = cnn_out_channels * 2

        elif temporal_model_type == "transformer":
            self.temporal_model = TransformerLayer.TransformerLayer(
                input_size=self.cnn_out_channels,
                hidden_size=self.cnn_out_channels * 2,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
            temporal_output_size = cnn_out_channels * 2

        else:
            raise ValueError(f"Unknown temporal_model_type: {temporal_model_type}")

        self.loss = dict()
        self.criterion_init()
        # 損失の重みを調整 - Loss爆発とWERを改善するための設定
        self.loss_weights = {
            "ConvCTC": 0.8,  # CNNからの予測を重視してWER改善
            "SeqCTC": 1.2,  # BiLSTMからの予測をより重視
            "Dist": 0.15,  # 蒸留損失を大幅に削減してLoss爆発を防ぐ
        }

        # 相関学習モジュールを追加
        self.spatial_correlation = SpatialCorrelationModule(input_dim=cnn_out_channels)

        # クラス分類用の線形層
        self.classifier = nn.Linear(temporal_output_size, num_classes)

        self._initialize_weights()

        gloss_dict = {word: [i] for i, word in enumerate(vocabulary)}

        # ログソフトマックス（CTC損失用）
        # 言語モデルを使用するgreedyデコーダーを使用
        self.decoder = decode.Decode(
            gloss_dict=gloss_dict,
            num_classes=num_classes,
            search_mode="greedy",  # 言語モデル統合型のグリーディサーチを使用
            blank_id=self.blank_id,
        )


    def _initialize_weights(self):
        """
        重みの初期化を強化した関数 - WER改善のためブランクトークンバイアスを大幅に改善
        """
        # Classifier層の初期化
        nn.init.xavier_uniform_(self.classifier.weight)

        # バイアスの特別な初期化 - ブランクを大幅に抑制し、他のクラスを強調
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.1)
            with torch.no_grad():
                self.classifier.bias.data[0] = -2.0 


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
        input_lengths,
        target_lengths,
        mode,
    ):
        """
        src_feature:[batch, C, T, J]
        tgt_feature:[batch, max_len]答えをバッチサイズの最大値に合わせてpaddingしている
        input_lengths:[batch]1D CNNに通した後のデータ
        target_lengths:[batch]tgt_featureで最大値に伸ばしたので本当の長さ
        current_epoch: 現在のエポック番号（グラフ作成時に使用）
        """
        
        N, C, T, J = src_feature.shape
        src_feature = src_feature.permute(0, 3, 1, 2).contiguous().view(N, C * J, T)
        spatial_feature = spatial_feature.permute(0, 2, 1)

        if torch.isnan(src_feature).any():
            print("警告: 入力src_featureにNaN値が検出されました")
            exit(1)
        # 無限大の値があるかチェック
        if torch.isinf(src_feature).any():
            print("警告: 入力src_featureに無限大の値が検出されました")

        # CNNモデルの実行
        logging.info("=== CNNモデル処理開始 ===")

        if self.cnn_model_type == "DualCNNWithCTC":
            cnn_out, cnn_logit, updated_lgt = self.cnn_model(
                skeleton_feat=src_feature, hand_feat=spatial_feature, lgt=input_lengths
            )
        else:
            cnn_out, cnn_logit, updated_lgt = self.cnn_model(
                skeleton_feat=src_feature,
                spatial_feature=spatial_feature,
                lgt=input_lengths,
            )

        logging.info(f"CNN出力完了 - logits範囲: {cnn_logit.min():.2f}~{cnn_logit.max():.2f}")

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
        logging.info("=== 時系列モデル処理開始 ===")
        tm_outputs = self.temporal_model(cnn_out, updated_lgt)
        predictions = tm_outputs["predictions"]
        
        # 分類器の実行
        outputs = self.classifier(predictions)
        
        logging.info(f"最終出力形状: {outputs.shape}, 範囲: {outputs.min():.2f}~{outputs.max():.2f}")

       
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
        
        # ロジットとラベル情報の取得
        conv_logits = ret_dict["conv_logits"]
        seq_logits = ret_dict["sequence_logits"]
        feat_len = ret_dict["feat_len"]
        
        logging.info("=== CTC損失計算開始 ===")
        
        # 損失計算
        for k, weight in self.loss_weights.items():
            if k == "ConvCTC":
                try:
                    # CTC損失計算の前にlogitsの健全性をチェック
                    logits_std = conv_logits.std()
                    logits_max_abs = conv_logits.abs().max()
                    
                    ctc_loss = self.loss["CTCLoss"](
                        conv_logits.log_softmax(-1),
                        label.cpu().int(),
                        feat_len.cpu().int(),
                        label_lgt.cpu().int(),
                    )
                    
                    # zero_infinityで0になった場合の補正
                    if torch.any(ctc_loss == 0.0) and (logits_std > 3.0 or logits_max_abs > 8.0):
                        # 極端なlogitsが原因で0になった場合、ペナルティを課す
                        logits_penalty = torch.clamp(logits_max_abs - 5.0, min=0.0) ** 2
                        ctc_loss = ctc_loss + logits_penalty * 0.1
                        logging.warning(f"極端なlogits検出: std={logits_std:.2f}, max_abs={logits_max_abs:.2f}, ペナルティ追加")
                    
                    conv_loss += weight * ctc_loss.mean()
                    
                except Exception as e:
                    logging.error(f"ConvCTC損失計算エラー: {e}")
                    conv_loss = torch.tensor(0.0, device=conv_logits.device, requires_grad=True)
                    
            elif k == "SeqCTC":
                try:
                    # SeqCTC損失でも同様の補正を適用
                    seq_logits_std = seq_logits.std()
                    seq_logits_max_abs = seq_logits.abs().max()
                    
                    ctc_loss = self.loss["CTCLoss"](
                        seq_logits.log_softmax(-1),
                        label.cpu().int(),
                        feat_len.cpu().int(),
                        label_lgt.cpu().int(),
                    )
                    
                    # zero_infinityで0になった場合の補正
                    if torch.any(ctc_loss == 0.0) and (seq_logits_std > 3.0 or seq_logits_max_abs > 8.0):
                        logits_penalty = torch.clamp(seq_logits_max_abs - 5.0, min=0.0) ** 2
                        ctc_loss = ctc_loss + logits_penalty * 0.1
                        logging.warning(f"Seq極端なlogits検出: std={seq_logits_std:.2f}, max_abs={seq_logits_max_abs:.2f}")
                    
                    seq_loss += weight * ctc_loss.mean()
                    
                except Exception as e:
                    logging.error(f"SeqCTC損失計算エラー: {e}")
                    seq_loss = torch.tensor(0.0, device=seq_logits.device, requires_grad=True)
                    
            elif k == "Dist":
                try:
                
                    dist_loss += weight * self.loss["distillation"](
                        conv_logits,
                        seq_logits.detach(),
                        use_blank=False,
                    )
                   
                except Exception as e:
                    logging.error(f"蒸留損失計算エラー: {e}")
                    dist_loss = torch.tensor(0.0, device=conv_logits.device, requires_grad=True)
        
        loss = conv_loss + seq_loss + dist_loss
        logging.info(f"損失: Conv={conv_loss.item():.3f}, Seq={seq_loss.item():.3f}, Dist={dist_loss.item():.3f}, 総計={loss.item():.3f}")
        return loss

    def criterion_init(self):
        # zero_infinity=Trueに変更してinf値を0に置換
        # blank_indexを明示的に指定してブランクトークンの扱いを最適化
        self.loss["CTCLoss"] = torch.nn.CTCLoss(
            blank=self.blank_id, 
            reduction="none", 
            zero_infinity=True
        )
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
