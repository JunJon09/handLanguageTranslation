import os
import time
import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):

        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        # 標準的なCTC損失関数を初期化
        self.ctc_loss = torch.nn.CTCLoss(
            blank=blank_id, reduction="mean", zero_infinity=True
        )

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.GreedySearch(nn_output, vid_lgt, probs)

    def GreedySearch(self, nn_output, vid_lgt, probs=False):
        """
        グリーディサーチによるCTC復号
        - 入力: nn_output (B, T, N), ソフトマックスを通すべき出力
        - 出力: デコード結果のリスト
        """
        # 確率に変換（必要であれば）
        if not probs:
            nn_output = nn_output.softmax(-1)

        # 最も確率の高いクラスを選択
        pred_labels = torch.argmax(nn_output, dim=2)  # (B, T)

        ret_list = []
        for batch_idx, length in enumerate(vid_lgt):
            # 有効な長さだけを使用
            sequence = pred_labels[batch_idx, :length]

            # 1. 連続する同じラベルを除去
            grouped = [x[0] for x in groupby(sequence.cpu().numpy())]

            # 2. ブランクを除去
            filtered = [label for label in grouped if label != self.blank_id]

            # 3. インデックスを単語に変換
            decoded = [
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(filtered)
            ]
            ret_list.append(decoded)

        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [
                x[0] for x in groupby(index_list[batch_idx][: vid_lgt[batch_idx]])
            ]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append(
                [
                    (self.i2g_dict[int(gloss_id)], idx)
                    for idx, gloss_id in enumerate(max_result)
                ]
            )
        return ret_list

    def calculate_loss(self, log_probs, targets, input_lengths, target_lengths):
        """
        CTC損失を計算する
        - log_probs: 対数確率 (T, B, C)
        - targets: ターゲットシーケンス (B, S)
        - input_lengths: 入力の長さ (B,)
        - target_lengths: ターゲットの長さ (B,)
        """
        # 入力形式を(T, B, C)に変換
        if log_probs.size(0) != log_probs.size(1):  # (B, T, C)の場合
            log_probs = log_probs.permute(1, 0, 2)

        # CTC損失の計算
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        return loss
