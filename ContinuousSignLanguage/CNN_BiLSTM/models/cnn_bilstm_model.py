import CNN_BiLSTM.models.one_dcnn as cnn
import CNN_BiLSTM.models.BiLSTM as BiLSTM
import CNN_BiLSTM.models.decode as decode
import CNN_BiLSTM.models.criterions as criterions

from torch import nn
import torch
import torch.nn.functional as F

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
        blank_idx=1,
    ):
        super().__init__()

        # blank_idxをクラス変数として保存
        self.blank_id = blank_idx

        # 1DCNNモデル
        self.cnn_model = cnn.DualCNNWithCTC(
            skeleton_input_size=in_channels,
            hand_feature_size=50,
            skeleton_hidden_size=128,
            hand_hidden_size=128,
            fusion_hidden_size=192,
            num_classes=num_classes,
            blank_idx=blank_idx,
        ).to("cpu")

        self.loss = dict()
        self.criterion_init()
        self.loss_weights = {
            'ConvCTC': 1.0,  # ConvCTCの損失に対する重み
            'SeqCTC': 0.5,   # SeqCTCの損失に対する重み
            'Dist': 0.1      # 知識蒸留（Distillation）損失に対する重み
        }

        self.temporal_model = BiLSTM.BiLSTMLayer(
            rnn_type="LSTM",
            input_size=192,
            hidden_size=192,  # 隠れ層のサイズ
            num_layers=2,
            bidirectional=True,
        )

        # クラス分類用の線形層
        self.classifier = nn.Linear(192, num_classes)

        # 重みの初期化を改善
        # self._initialize_weights()

        # ログソフトマックス（CTC損失用）
        gloss_dict = {'<blank>': [0], 'i': [1], 'you': [2], 'he': [3], 'teacher': [4], 'friend': [5], 'today': [6], 'tomorrow': [7], 'school': [8], 'hospital': [9], 'station': [10], 'go': [11], 'come': [12], 'drink': [13], 'like': [14], 'dislike': [15], 'busy': [16], 'use': [17], 'meet': [18], 'where': [19], 'question': [20], 'pp': [21], 'with': [22], 'water': [23], 'food': [24], 'money': [25], 'apple': [26], 'banana': [27], '<pad>': [28]}
        self.decoder = decode.Decode(gloss_dict=gloss_dict, num_classes=num_classes, search_mode="max", blank_id=blank_idx)

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
        # 数値安定性のための正規化を追加
        # src_feature形状は[batch, C*J, T]なので、dim=2ではなくdim=1に沿って正規化
        # eps = 1e-5
        # src_mean = src_feature.mean(dim=1, keepdim=True)
        # src_std = src_feature.std(dim=1, keepdim=True) + eps
        # src_feature = (src_feature - src_mean) / src_std

        # # 大きすぎる値や小さすぎる値をクリップ
        # src_feature = torch.clamp(src_feature, min=-5.0, max=5.0)
        # print(src_feature.shape, "src_feature.shape")
        # cnn_out = self.cnn_model(src_feature)  # return [batch, 512, T']
        cnn_out, cnn_logit, updated_lgt = self.cnn_model(
            skeleton_feat=src_feature, hand_feat=spatial_feature, lgt=input_lengths
        )
        print(cnn_out.shape, "cnn_out.shape", src_feature.shape, "src_feature.shape", input_lengths[0].shape, "input_lengths.shape", input_lengths[1].shape)

        if torch.isnan(cnn_out).any():
            print("層1の出力にNaN値が検出されました")
            # 問題のある入力値の確認
            problem_indices = torch.where(torch.isnan(cnn_out))
            print(
                f"問題がある入力の要素: {cnn_out[problem_indices[0][0], problem_indices[1][0], problem_indices[2][0]]}"
            )
            exit(1)

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

                else:  # シーケンスが長くなった場合はパディング
                    padding_length = cnn_seq_len - original_seq_len
                    padding = torch.zeros(
                        src_padding_mask.size(0),
                        padding_length,
                        dtype=src_padding_mask.dtype,
                        device=src_padding_mask.device,
                    )
                    src_padding_mask = torch.cat((src_padding_mask, padding), dim=1)


        # # クラス分類
        print(f"updated_lgt: {updated_lgt}")
        print(f"updated_lgtの形状: {updated_lgt.shape}")
        print(f"updated_lgtの値: {updated_lgt.tolist()}")
        tm_outputs = self.temporal_model(cnn_out, updated_lgt)
        print(tm_outputs["predictions"].shape, "tm_outputs['predictions'].shape")
        outputs = self.classifier(tm_outputs["predictions"])  # (batch, T', num_classes)
       

        pred = self.decoder.decode(outputs, updated_lgt, batch_first=False, probs=False)
        conv_pred = self.decoder.decode(cnn_logit, updated_lgt, batch_first=False, probs=False)
        print(pred, "pred")
        print(conv_pred, "conv_pred")
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
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            # elif k == 'Dist':
            #     loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
            #                                                ret_dict["sequence_logits"].detach(),
            #                                                use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = criterions.SeqKD(T=8)
        return self.loss
