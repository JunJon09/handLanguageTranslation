import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F
import logging


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
        日本手話のSOV語順構造を考慮した言語モデルを初期化
        日本手話: 主語(S) - 目的語(O) - 動詞(V) の語順
        """
        # 日本手話の語順パターンに基づく共起確率辞書
        # 品詞分類に基づく確率設定
        self.word_categories = {
            # 主語 (Subject) - 人称代名詞・人物
            'subject': [1, 2, 4, 5],  # 'i', 'you', 'teacher', 'friend'
            # 目的語 (Object) - 場所・物・概念
            'object': [6, 7, 8, 9, 10, 16, 26],  # 'today', 'tomorrow', 'school', 'hospital', 'station', など
            # 動詞 (Verb) - 動作・状態
            'verb': [11, 13, 14, 15, 21, 23],  # 'go', 'come', 'like', 'dislike', など
        }
        
        # SOV語順に基づく共起確率
        self.word_cooccur = {
            # S -> O パターン（主語の後に目的語が来る確率）
            (1, 6): 0.9,   # 'i' -> 'today' (私は今日)
            (1, 7): 0.9,   # 'i' -> 'tomorrow' (私は明日)
            (1, 8): 0.8,   # 'i' -> 'school' (私は学校)
            (1, 9): 0.7,   # 'i' -> 'hospital' (私は病院)
            (1, 26): 0.8,  # 'i' -> 場所/物
            (2, 8): 0.8,   # 'you' -> 'school' (あなたは学校)
            (2, 9): 0.7,   # 'you' -> 'hospital' (あなたは病院)
            (4, 8): 0.9,   # 'teacher' -> 'school' (先生は学校)
            
            # O -> V パターン（目的語の後に動詞が来る確率）
            (6, 11): 0.95,  # 'today' -> 'go' (今日行く)
            (7, 11): 0.95,  # 'tomorrow' -> 'go' (明日行く)
            (8, 11): 0.9,   # 'school' -> 'go' (学校に行く)
            (9, 11): 0.9,   # 'hospital' -> 'go' (病院に行く)
            (10, 11): 0.9,  # 'station' -> 'go' (駅に行く)
            (16, 14): 0.8,  # 対象 -> 'like' (〜が好き)
            (16, 15): 0.7,  # 対象 -> 'dislike' (〜が嫌い)
            (26, 14): 0.8,  # 対象 -> 'like'
            
            # S -> V パターン（2語文: 主語 + 動詞）
            (1, 11): 0.7,   # 'i' -> 'go' (私は行く)
            (1, 14): 0.8,   # 'i' -> 'like' (私は好き)
            (2, 11): 0.7,   # 'you' -> 'go' (あなたは行く)
            (2, 14): 0.8,   # 'you' -> 'like' (あなたは好き)
            
            # 動詞の後に来るべきでない単語（語順違反のペナルティ）
            (11, 1): 0.1,   # 'go' -> 'i' (動詞の後に主語は来ない)
            (11, 2): 0.1,   # 'go' -> 'you'
            (14, 1): 0.1,   # 'like' -> 'i'
            (14, 2): 0.1,   # 'like' -> 'you'
            
            # 連続する同じ品詞の制約
            (1, 2): 0.2,    # 主語の連続を制約
            (1, 4): 0.2,    # 'i' -> 'teacher'
            (11, 13): 0.3,  # 動詞の連続を制約
            (11, 14): 0.3,  # 'go' -> 'like'
        }
        
        # 品詞情報に基づくペナルティ/ボーナス
        self.pos_transition_scores = {
            ('subject', 'object'): 0.8,   # S -> O は高確率
            ('subject', 'verb'): 0.7,     # S -> V も許可（2語文）
            ('object', 'verb'): 0.9,      # O -> V は最高確率
            ('verb', 'subject'): 0.1,     # V -> S は低確率（語順違反）
            ('verb', 'object'): 0.2,      # V -> O は低確率（語順違反）
            ('verb', 'verb'): 0.3,        # V -> V は制約
        }

        # デフォルトスコア（辞書に存在しないペア用）
        self.default_score = 0.15
        
        # 語順違反に対するペナルティ重み
        self.sov_penalty_weight = 0.3

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


            # 1. 連続する同じラベルを除去
            grouped = [x[0] for x in groupby(sequence.cpu().numpy())]

            # 2. ブランクを完全に除去
            filtered = [label for label in grouped if label != self.blank_id]

            # 3. もしブランクが残っていないか確認して警告
            if self.blank_id in filtered:
                # 念のため再フィルタリング
                filtered = [label for label in filtered if label != self.blank_id]

            # 4. 出力が空の場合は特別処理（最頻出の非ブランクトークンを選択）
            if len(filtered) == 0:
                # ブランク以外で最も確率の高いクラスを取得
                mean_probs = nn_output[batch_idx, :length].mean(dim=0)
                blank_mask = torch.ones_like(mean_probs)
                blank_mask[self.blank_id] = 0  # ブランクを除外
                masked_probs = mean_probs * blank_mask
                best_non_blank = torch.argmax(masked_probs).item()
                filtered = [best_non_blank]

            # 5. インデックスを単語に変換
            decoded = []
            for idx, gloss_id in enumerate(filtered):
                if gloss_id in self.i2g_dict:
                    decoded.append((self.i2g_dict[int(gloss_id)], idx))
                else:
                    print(f"警告: 未知のグロスID {gloss_id}")

            logging.info(f"デコード結果: {decoded}")
            ret_list.append(decoded)

        return ret_list

    def get_word_category(self, word_id):
        """単語IDから品詞カテゴリを取得"""
        for category, word_list in self.word_categories.items():
            if word_id in word_list:
                return category
        return 'unknown'
    
    def calculate_sov_score(self, prev_token, current_token, sequence_so_far):
        """
        SOV語順構造を考慮したスコアを計算
        """
        if prev_token is None:
            return 0.5  # 文頭のデフォルトスコア
        
        prev_category = self.get_word_category(prev_token)
        current_category = self.get_word_category(current_token)
        
        # 基本的な品詞遷移スコア
        transition_key = (prev_category, current_category)
        pos_score = self.pos_transition_scores.get(transition_key, 0.3)
        
        # 語順違反チェック
        sequence_categories = [self.get_word_category(token) for token in sequence_so_far]
        
        # SOV語順違反のペナルティ
        penalty = 0
        if current_category == 'subject' and 'verb' in sequence_categories:
            # 動詞の後に主語が来る = 語順違反
            penalty += self.sov_penalty_weight
        elif current_category == 'object' and 'verb' in sequence_categories:
            # 動詞の後に目的語が来る = 語順違反
            penalty += self.sov_penalty_weight * 0.8
        
        # 動詞の重複チェック
        if current_category == 'verb' and sequence_categories.count('verb') >= 1:
            penalty += 0.2  # 動詞の重複にペナルティ
        
        return max(0.05, pos_score - penalty)
    def GreedySearchWithLM(self, nn_output, vid_lgt, probs=False):
        """
        日本手話のSOV語順を考慮した言語モデル付きグリーディサーチ
        """
        # 確率に変換（必要であれば）
        if not probs:
            nn_output = nn_output.softmax(-1)

        ret_list = []
        for batch_idx, length in enumerate(vid_lgt):
            # 有効な長さだけを使用
            sequence_probs = nn_output[batch_idx, :length]

            # SOV語順を考慮したデコード
            prev_token = None
            filtered_tokens = []
            sequence_so_far = []

            # フレームごとに最も確率の高いトークンを取得
            for t in range(length):
                frame_probs = sequence_probs[t]

                # ブランクを除く上位5トークン候補を取得（SOV考慮のため候補数増加）
                topk = min(5, self.num_classes - 1)
                values, indices = torch.topk(frame_probs, k=topk + 1)

                # ブランクを除外
                non_blank_values = []
                non_blank_indices = []
                for i, idx in enumerate(indices):
                    if idx.item() != self.blank_id:
                        non_blank_values.append(values[i].item())
                        non_blank_indices.append(idx.item())

                if len(non_blank_indices) == 0:
                    continue

                # SOV語順を考慮したスコアを計算
                sov_scores = []
                for i, token_id in enumerate(non_blank_indices):
                    # CTC確率スコア
                    ctc_score = non_blank_values[i]

                    # 単語共起スコア
                    cooccur_score = 0
                    if prev_token is not None:
                        cooccur_score = self.word_cooccur.get(
                            (prev_token, token_id), self.default_score
                        )

                    # SOV語順スコア
                    sov_score = self.calculate_sov_score(prev_token, token_id, sequence_so_far)

                    # 最終スコア = CTC確率 * α + 共起確率 * β + SOV語順スコア * γ
                    final_score = (
                        ctc_score * 0.5 +           # CTC確率の重み
                        cooccur_score * 0.3 +       # 共起確率の重み
                        sov_score * 0.2             # SOV語順の重み
                    )
                    sov_scores.append(final_score)

                # 最も高いスコアを持つトークンを選択
                if sov_scores:
                    best_idx = np.argmax(sov_scores)
                    best_token = non_blank_indices[best_idx]

                    # 前のトークンと異なる場合のみ追加（連続する同じトークンを除去）
                    if best_token != prev_token:
                        filtered_tokens.append(best_token)
                        sequence_so_far.append(best_token)
                        prev_token = best_token
                        
                        # デバッグ情報
                        best_category = self.get_word_category(best_token)
                        logging.info(f"選択されたトークン: {best_token} (品詞: {best_category}, スコア: {sov_scores[best_idx]:.3f})")

            # 2. SOV語順の後処理：語順を最適化
            filtered_tokens = self.reorder_sov_sequence(filtered_tokens)

            # 3. 結果がなければフォールバック
            if not filtered_tokens:
                # フォールバック: 標準のグリーディサーチ
                pred_labels = torch.argmax(sequence_probs, dim=1)
                grouped = [x[0] for x in groupby(pred_labels.cpu().numpy())]
                filtered_tokens = [label for label in grouped if label != self.blank_id]

            # 4. インデックスを単語に変換
            decoded = [
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(filtered_tokens)
                if gloss_id in self.i2g_dict
            ]

            # SOV語順の分析情報をログ出力
            categories = [self.get_word_category(token) for token in filtered_tokens]
            logging.info(f"SOV語順分析（バッチ {batch_idx}）: {categories}")
            logging.info(f"SOVデコード結果（バッチ {batch_idx}）: {decoded}")
            ret_list.append(decoded)

        return ret_list

    def reorder_sov_sequence(self, tokens):
        """
        SOV語順に従ってトークンシーケンスを並べ替え
        """
        if len(tokens) <= 2:
            return tokens  # 短いシーケンスはそのまま
        
        # 品詞別にトークンを分類
        subjects = []
        objects = []
        verbs = []
        others = []
        
        for token in tokens:
            category = self.get_word_category(token)
            if category == 'subject':
                subjects.append(token)
            elif category == 'object':
                objects.append(token)
            elif category == 'verb':
                verbs.append(token)
            else:
                others.append(token)
        
        # SOV語順で再構築
        reordered = []
        
        # 主語を先頭に配置
        reordered.extend(subjects)
        
        # その他の要素を配置
        reordered.extend(others)
        
        # 目的語を配置
        reordered.extend(objects)
        
        # 動詞を最後に配置
        reordered.extend(verbs)
        
        # 空でない場合のみ返す
        if reordered:
            logging.info(f"SOV語順並べ替え: {tokens} -> {reordered}")
            return reordered
        else:
            return tokens  # 並べ替えに失敗した場合は元のまま

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
