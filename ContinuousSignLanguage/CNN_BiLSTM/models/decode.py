import os
import time
import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):

        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id

        # 分析用のデータ蓄積
        self.token_frequencies = []  # 各バッチの非ブランクトークン頻度を記録
        self.batch_count = 0
        self.plot_dir = os.path.join(os.getcwd(), "plots", "token_analysis")
        os.makedirs(self.plot_dir, exist_ok=True)

        # 単語の共起関係に基づく簡易言語モデルを初期化
        self.init_language_model()

        # デコードの設定
        self.lm_weight = 0.3  # 言語モデルの重み
        self.use_language_model = True  # 言語モデルを使用するフラグ

        # 標準的なCTC損失関数を初期化
        self.ctc_loss = torch.nn.CTCLoss(
            blank=blank_id, reduction="mean", zero_infinity=True
        )
        self.count = 0

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

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False, analyze=False):
        """
        デコード処理を行う

        Args:
            nn_output: ネットワークの出力
            vid_lgt: 各ビデオの有効長さ
            batch_first: バッチ次元が最初かどうか
            probs: 入力がすでに確率になっているかどうか
            analyze: トークン分布を分析するかどうか
        """
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)

        # 分析モードがオンの場合、トークン分布を分析
       
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

            # 1. 連続する同じラベルを除去
            grouped = [x[0] for x in groupby(sequence.cpu().numpy())]

            # 2. ブランクを完全に除去
            filtered = [label for label in grouped if label != self.blank_id]

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

    def analyze_frame_by_frame_distribution(self, nn_output, vid_lgt, save_plot=True):
        """
        バッチサイズ1を前提として、各フレームごとの非ブランクトークンTop5をプロットする
        """
        self.count += 1
        if not isinstance(nn_output, torch.Tensor):
            print("警告: 入力がTensorではありません")
            return []

        # ソフトマックス適用（必要な場合）
       
        probs = torch.softmax(nn_output, dim=-1)

        # バッチサイズが1であることを確認
        if probs.size(0) != 1:
            print(f"警告: バッチサイズが1ではありません (現在: {probs.size(0)})")
            return []

        batch_idx = 0
        length = vid_lgt[batch_idx]

        # 1つのサンプルの全フレームを処理
        sequence_probs = probs[batch_idx, :length]  # (T, num_classes)

        # ブランクを除外したマスク
        non_blank_mask = torch.ones(
            self.num_classes, dtype=torch.bool, device=probs.device
        )
        non_blank_mask[self.blank_id] = False

        # 各フレームでのTop5非ブランクトークンを記録
        frame_top5_data = {"frames": [], "token_names": [], "probabilities": []}

        # 全フレームで出現するトークンを収集（グラフの一貫性のため）
        all_tokens_set = set()

        for frame_idx in range(length):
            frame_probs = sequence_probs[frame_idx]  # (num_classes,)

            # 非ブランクトークンの確率を取得
            non_blank_probs = frame_probs[non_blank_mask]
            non_blank_ids = torch.arange(self.num_classes, device=probs.device)[
                non_blank_mask
            ]

            # Top5を取得
            top5_values, top5_indices = torch.topk(
                non_blank_probs, min(5, len(non_blank_probs))
            )

            # フレームデータを記録
            frame_tokens = []
            frame_probs_list = []

            for i in range(len(top5_indices)):
                token_idx = non_blank_ids[top5_indices[i]].item()
                token_name = self.i2g_dict.get(token_idx, f"unknown_{token_idx}")
                token_prob = top5_values[i].item()

                frame_tokens.append(token_name)
                frame_probs_list.append(token_prob)
                all_tokens_set.add(token_name)


            frame_top5_data["frames"].append(frame_idx)
            frame_top5_data["token_names"].append(frame_tokens)
            frame_top5_data["probabilities"].append(frame_probs_list)

        # グラフを作成
        if save_plot:
            print(self.count)
            self.plot_frame_by_frame_trends(frame_top5_data, all_tokens_set, length)

        return frame_top5_data

    def plot_frame_by_frame_trends(self, frame_data, all_tokens, num_frames):
        """
        フレームごとのTop5トークン推移を折線グラフでプロット
        """
        # 全フレームで出現した上位トークンを特定（頻度順）
        token_frequency = Counter()
        for frame_tokens in frame_data["token_names"]:
            for token in frame_tokens:
                token_frequency[token] += 1

        # 最頻出の上位8トークンをプロット対象とする
        top_tokens = [token for token, _ in token_frequency.most_common(8)]

        # 各トークンのフレームごと確率を整理
        token_trends = {token: [0.0] * num_frames for token in top_tokens}

        for frame_idx, (frame_tokens, frame_probs) in enumerate(
            zip(frame_data["token_names"], frame_data["probabilities"])
        ):
            for token, prob in zip(frame_tokens, frame_probs):
                if token in token_trends:
                    token_trends[token][frame_idx] = prob

        # グラフの作成
        plt.figure(figsize=(15, 10))

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]
        markers = ["o", "s", "^", "D", "v", "p", "*", "h"]

        frames = list(range(num_frames))

        for i, token in enumerate(top_tokens):
            plt.plot(
                frames,
                token_trends[token],
                label=f"{token}",
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                markersize=4,
                linewidth=1.5,
                alpha=0.8,
            )

        plt.xlabel("frame", fontsize=12)
        plt.ylabel("probability", fontsize=12)
        plt.title(
            "Probability transition per frame", fontsize=14, fontweight="bold"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 統計情報を表示
        print(f"\n=== フレームごとトークン分析統計 ===")
        for i, token in enumerate(top_tokens):
            max_prob = max(token_trends[token])
            avg_prob = np.mean([p for p in token_trends[token] if p > 0])
            appearance_count = sum(1 for p in token_trends[token] if p > 0)

          
        # ファイル保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.plot_dir, f"frame_by_frame_top5_{self.count}.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nフレームごとグラフを保存しました: {save_path}")

        plt.close()

    def analyze_frame_level_patterns(self, nn_output, vid_lgt, save_plot=True):
        """
        フレームレベルでのパターン分析（ピーク検出、変化点検出など）
        """
        frame_data = self.analyze_frame_by_frame_distribution(
            nn_output, vid_lgt, save_plot=False
        )

        if not frame_data["frames"]:
            return

        print(f"\n=== フレームレベルパターン分析 ===")

        # 各フレームでのTop1トークンを抽出
        top1_sequence = []
        for frame_tokens in frame_data["token_names"]:
            if frame_tokens:
                top1_sequence.append(frame_tokens[0])
            else:
                top1_sequence.append("none")

        # トークンの変化点を検出
        change_points = []
        prev_token = None

        for i, token in enumerate(top1_sequence):
            if token != prev_token and prev_token is not None:
                change_points.append(i)
                print(f"フレーム {i}: {prev_token} → {token}")
            prev_token = token

        print(f"検出された変化点数: {len(change_points)}")
        print(
            f"平均的な手話単語の長さ: {len(top1_sequence) / (len(change_points) + 1):.1f} フレーム"
        )

        # 最も安定したトークン（連続出現が長い）を検出
        from itertools import groupby

        grouped_tokens = [(k, len(list(g))) for k, g in groupby(top1_sequence)]
        stable_tokens = sorted(grouped_tokens, key=lambda x: x[1], reverse=True)[:3]

        print(f"\n最も安定したトークン（連続出現長順）:")
        for token, length in stable_tokens:
            print(f"  {token}: {length}フレーム連続")

        if save_plot:
            self.plot_frame_by_frame_trends(
                frame_data, set(top1_sequence), len(top1_sequence)
            )

