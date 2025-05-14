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

        # 単語の共起関係に基づく簡易言語モデルを初期化
        self.init_language_model()

        # デコードの設定
        self.lm_weight = 0.3  # 言語モデルの重み
        self.use_language_model = True  # 言語モデルを使用するフラグ

        # 標準的なCTC損失関数を初期化
        self.ctc_loss = torch.nn.CTCLoss(
            blank=blank_id, reduction="mean", zero_infinity=True
        )

    def init_language_model(self):
        """
        単語の共起確率に基づく簡易言語モデルを初期化
        実際のアプリケーションでは、この部分を外部の言語モデルに置き換えることも可能
        """
        # 単語の共起確率辞書（実際にはより複雑なモデルが必要）
        # キー: (前の単語ID, 現在の単語候補ID), 値: スコア
        # 高頻度な手話単語シーケンスに対して高いスコアを与える
        self.word_cooccur = {
            # 'i' の後によく続く単語
            (1, 6): 0.8,  # 'i' -> 'go' (高確率)
            (1, 7): 0.7,  # 'i' -> 'come'
            (1, 14): 0.7,  # 'i' -> 'like'
            (1, 15): 0.5,  # 'i' -> 'dislike'
            # 'you' の後によく続く単語
            (2, 14): 0.8,  # 'you' -> 'like'
            (2, 11): 0.7,  # 'you' -> 'go'
            # 場所名詞の後によく続く単語
            (8, 11): 0.9,  # 'school' -> 'go'
            (9, 11): 0.9,  # 'hospital' -> 'go'
            (10, 11): 0.9,  # 'station' -> 'go'
            # 他の一般的な単語のペア
            (4, 5): 0.6,  # 'teacher' -> 'friend'
            (6, 8): 0.5,  # 'today' -> 'school'
            (7, 8): 0.5,  # 'tomorrow' -> 'school'
        }

        # デフォルトスコア（辞書に存在しないペア用）
        self.default_score = 0.1

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        elif self.search_mode == "greedy_lm" and self.use_language_model:
            return self.GreedySearchWithLM(nn_output, vid_lgt, probs)
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

            print(f"バッチ {batch_idx} のシーケンス長: {length}")
            print(f"デコード前のシーケンス: {sequence.cpu().numpy()}")

            # 1. 連続する同じラベルを除去
            grouped = [x[0] for x in groupby(sequence.cpu().numpy())]
            print(f"グループ化後: {grouped}")

            # 2. ブランクを完全に除去
            filtered = [label for label in grouped if label != self.blank_id]
            print(f"ブランク除去後: {filtered}")

            # 3. もしブランクが残っていないか確認して警告
            if self.blank_id in filtered:
                print(f"警告: ブランク除去後もブランク({self.blank_id})が残っています")
                # 念のため再フィルタリング
                filtered = [label for label in filtered if label != self.blank_id]
                print(f"再フィルタリング後: {filtered}")

            # 4. 出力が空の場合は特別処理（最頻出の非ブランクトークンを選択）
            if len(filtered) == 0:
                print("出力が空です - 非ブランクトークンで最も確率が高いものを選択")
                # ブランク以外で最も確率の高いクラスを取得
                mean_probs = nn_output[batch_idx, :length].mean(dim=0)
                blank_mask = torch.ones_like(mean_probs)
                blank_mask[self.blank_id] = 0  # ブランクを除外
                masked_probs = mean_probs * blank_mask
                best_non_blank = torch.argmax(masked_probs).item()
                filtered = [best_non_blank]
                print(f"選択された非ブランクトークン: {best_non_blank}")

            # 5. インデックスを単語に変換
            decoded = []
            for idx, gloss_id in enumerate(filtered):
                if gloss_id in self.i2g_dict:
                    decoded.append((self.i2g_dict[int(gloss_id)], idx))
                else:
                    print(f"警告: 未知のグロスID {gloss_id}")

            print(f"デコード結果: {decoded}")
            ret_list.append(decoded)

        return ret_list

    def GreedySearchWithLM(self, nn_output, vid_lgt, probs=False):
        """
        言語モデルを使用したグリーディサーチによるCTC復号
        """
        # 確率に変換（必要であれば）
        if not probs:
            nn_output = nn_output.softmax(-1)

        ret_list = []
        for batch_idx, length in enumerate(vid_lgt):
            # 有効な長さだけを使用
            sequence_probs = nn_output[batch_idx, :length]

            # 1. 連続したフレームをオーバーラップして処理
            prev_token = None
            filtered_tokens = []

            # フレームごとに最も確率の高いトークンを取得
            for t in range(length):
                frame_probs = sequence_probs[t]

                # ブランクを除く上位3トークン候補を取得
                topk = min(3, self.num_classes - 1)  # 最大でクラス数-1個を取得
                values, indices = torch.topk(frame_probs, k=topk + 1)  # ブランク用に+1

                # ブランクを除外
                non_blank_values = []
                non_blank_indices = []
                for i, idx in enumerate(indices):
                    if idx.item() != self.blank_id:
                        non_blank_values.append(values[i].item())
                        non_blank_indices.append(idx.item())

                if len(non_blank_indices) == 0:
                    continue

                # 言語モデルで補正したスコアを計算
                lm_scores = []
                for i, token_id in enumerate(non_blank_indices):
                    # CTC確率スコア
                    ctc_score = non_blank_values[i]

                    # 言語モデルスコア（前のトークンとの共起確率）
                    lm_score = 0
                    if prev_token is not None:
                        lm_score = self.word_cooccur.get(
                            (prev_token, token_id), self.default_score
                        )

                    # 最終スコア = CTC確率 * (1-α) + 言語モデル確率 * α
                    final_score = (
                        ctc_score * (1 - self.lm_weight) + lm_score * self.lm_weight
                    )
                    lm_scores.append(final_score)

                # 最も高いスコアを持つトークンを選択
                if lm_scores:
                    best_idx = np.argmax(lm_scores)
                    best_token = non_blank_indices[best_idx]

                    # 前のトークンと異なる場合のみ追加（連続する同じトークンを除去）
                    if best_token != prev_token:
                        filtered_tokens.append(best_token)
                        prev_token = best_token

            # 2. 結果がなければフォールバック
            if not filtered_tokens:
                # フォールバック: 標準のグリーディサーチ
                pred_labels = torch.argmax(sequence_probs, dim=1)
                grouped = [x[0] for x in groupby(pred_labels.cpu().numpy())]
                filtered_tokens = [label for label in grouped if label != self.blank_id]

            # 3. インデックスを単語に変換
            decoded = [
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(filtered_tokens)
                if gloss_id in self.i2g_dict
            ]

            print(f"LMありのデコード結果（バッチ {batch_idx}）: {decoded}")
            ret_list.append(decoded)

        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            # 有効な長さまでのシーケンスを抽出
            sequence = index_list[batch_idx][: vid_lgt[batch_idx]]

            print(
                f"MaxDecode - バッチ {batch_idx} のシーケンス長: {vid_lgt[batch_idx]}"
            )
            print(f"MaxDecode - デコード前のシーケンス: {sequence.cpu().numpy()}")

            # 1. 連続する同じラベルを除去
            group_result = [x[0] for x in groupby(sequence.cpu().numpy())]
            print(f"MaxDecode - グループ化後: {group_result}")

            # 2. ブランクを完全に除去
            filtered = [x for x in group_result if x != self.blank_id]
            print(f"MaxDecode - ブランク除去後: {filtered}")

            # 3. 出力が空の場合は特別処理
            if len(filtered) == 0:
                print(
                    "MaxDecode - 出力が空です - 非ブランクトークンで最も確率が高いものを選択"
                )
                # ブランク以外で最も確率の高いクラスを取得
                mean_probs = nn_output[batch_idx, : vid_lgt[batch_idx]].mean(dim=0)
                blank_mask = torch.ones_like(mean_probs)
                blank_mask[self.blank_id] = 0  # ブランクを除外
                masked_probs = mean_probs * blank_mask
                best_non_blank = torch.argmax(masked_probs).item()
                filtered = [best_non_blank]
                print(f"MaxDecode - 選択された非ブランクトークン: {best_non_blank}")

            # 4. 結果処理
            decoded = []
            for idx, gloss_id in enumerate(filtered):
                if gloss_id in self.i2g_dict:
                    decoded.append((self.i2g_dict[int(gloss_id)], idx))
                else:
                    print(f"MaxDecode - 警告: 未知のグロスID {gloss_id}")

            print(f"MaxDecode - デコード結果: {decoded}")
            ret_list.append(decoded)

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
