import torch
import time  # 時間スタンプ用にtimeモジュールを追加


def beam_search_decode(log_probs, beam_width=10, blank_id=1):
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
    num_classes = log_probs.size(2)
    decoded_sequences = []

    # デバッグ用
    print(f"ビームサーチデコード - blank_id: {blank_id}")
    print(f"log_probs形状: {log_probs.shape}")

    # 各クラスの確率分布を確認
    class_probs = torch.exp(log_probs).mean(dim=(0, 1))
    top5_probs, top5_indices = torch.topk(class_probs, 5)
    print(f"確率の高い上位5クラス: {top5_indices.tolist()}")
    print(f"上位5クラスの確率: {top5_probs.tolist()}")

    # フレームごとの確率変化をグラフ化（最初のバッチのみ）
    if batch_size > 0:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # 最初のバッチのデータを使用
            first_batch_probs = torch.exp(log_probs[:, 0, :]).cpu().numpy()
            time_steps = first_batch_probs.shape[0]

            # 上位5クラスのインデックスを取得
            top_classes = top5_indices.cpu().numpy()

            # グラフの準備
            plt.figure(figsize=(12, 8))

            # 上位5クラスの時間変化をプロット
            for i, class_idx in enumerate(top_classes):
                class_probs_over_time = first_batch_probs[:, class_idx]
                plt.plot(
                    range(time_steps),
                    class_probs_over_time,
                    label=f"Class {class_idx}",
                    linewidth=2,
                )

            # ブランクIDの時間変化を特別に表示（上位5に含まれていない場合）
            if blank_id not in top_classes:
                blank_probs_over_time = first_batch_probs[:, blank_id]
                plt.plot(
                    range(time_steps),
                    blank_probs_over_time,
                    label=f"Blank (ID:{blank_id})",
                    linewidth=2,
                    linestyle="--",
                    color="black",
                )

            # グラフの設定
            plt.xlabel("タイムステップ (フレーム)")
            plt.ylabel("確率")
            plt.title("フレームごとの各クラスの確率変化")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 確率の平均値も表示
            mean_probs = first_batch_probs.mean(axis=0)
            top5_mean_probs, top5_mean_indices = np.sort(mean_probs)[-5:], np.argsort(
                mean_probs
            )[-5:]

            # テキスト情報の追加
            plt.figtext(
                0.02,
                0.02,
                f"平均確率 Top5: クラス {top5_mean_indices.tolist()}\n"
                f"確率値: {[f'{p:.4f}' for p in top5_mean_probs.tolist()]}",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8),
            )

            # ファイルに保存
            import os

            os.makedirs("plots", exist_ok=True)
            plt.savefig(f"plots/frame_probabilities_{int(time.time())}.png")
            plt.close()

            print(f"フレームごとの確率グラフを 'plots/' ディレクトリに保存しました")

            # 確率ヒートマップも生成
            plt.figure(figsize=(14, 8))

            # 表示するクラス数を制限（例：上位10クラスと空白）
            top_n = min(10, num_classes - 1)
            overall_top_classes = np.argsort(mean_probs)[-top_n:]

            # ブランクIDを含める
            display_classes = list(overall_top_classes)
            if blank_id not in display_classes:
                display_classes.append(blank_id)
            display_classes.sort()

            # 選択したクラスのみのヒートマップを表示
            selected_probs = first_batch_probs[:, display_classes]

            # ヒートマップ
            plt.imshow(selected_probs.T, aspect="auto", cmap="viridis")
            plt.colorbar(label="確率")

            # 軸ラベル
            plt.xlabel("タイムステップ (フレーム)")
            plt.ylabel("クラス")
            plt.title("フレームごとのクラス確率ヒートマップ")

            # Y軸のクラスID表示
            plt.yticks(
                range(len(display_classes)),
                [
                    f"{idx}" + (" (Blank)" if idx == blank_id else "")
                    for idx in display_classes
                ],
            )

            # ヒートマップも保存
            plt.savefig(f"plots/probability_heatmap_{int(time.time())}.png")
            plt.close()

            print(f"確率ヒートマップも保存しました")

        except Exception as e:
            print(f"グラフ生成中にエラーが発生しました: {e}")
            # エラーが発生してもデコードは続行

    # フレームごとの最大確率クラスを確認（最初のバッチ）
    if batch_size > 0:
        first_batch_probs = torch.exp(log_probs[:, 0, :])
        max_probs, max_indices = torch.max(first_batch_probs, dim=1)

        # 各フレームでの最大確率クラスの統計
        unique_classes, counts = torch.unique(max_indices, return_counts=True)
        sorted_idx = torch.argsort(counts, descending=True)

        print("\nフレームごとの最大確率クラスの統計:")
        for i in range(min(5, len(unique_classes))):
            idx = sorted_idx[i]
            cls = unique_classes[idx].item()
            count = counts[idx].item()
            percent = count / len(max_indices) * 100
            print(
                f"クラス {cls}: {count}フレーム ({percent:.1f}%) - {'ブランク' if cls == blank_id else ''}"
            )

        # ブランク以外のクラスが予測される位置を確認
        non_blank_indices = (max_indices != blank_id).nonzero().squeeze().cpu().numpy()
        if len(non_blank_indices) > 0:
            print(
                f"\nブランク以外のクラスが最大確率となるフレーム位置: {non_blank_indices.tolist()[:10]}{'...' if len(non_blank_indices) > 10 else ''}"
            )
        else:
            print("\n警告: すべてのフレームでブランクが最大確率です")

    for b in range(batch_size):
        # 1バッチごとにビームサーチを実行
        seq_probs = log_probs[:, b, :]

        # バッチごとの確率分布も確認
        batch_probs = torch.exp(seq_probs).mean(dim=0)
        print(f"バッチ{b}の確率分布平均: {batch_probs.max().item():.4f} (最大値)")

        # ビーム初期化 - (前の文字列, 累積スコア, 最後の文字, 最後のタイムステップ)
        # 最後の文字と最後のタイムステップを追跡して、同じ文字の繰り返しを正しく処理
        beams = [([], 0.0, None, -1)]

        # 各タイムステップでビームを更新
        for t in range(seq_probs.size(0)):
            # 現在のタイムステップの確率
            probs = seq_probs[t]

            # 新しいビーム候補を格納するリスト
            new_beams = []

            # 上位k個のみを考慮するために、ソートされた確率とインデックスを取得
            topk_probs, topk_indices = torch.topk(
                probs, min(beam_width * 2, num_classes)
            )

            # 現在のビームを拡張
            for prefix, score, last_char, last_t in beams:
                # 上位k個の候補について処理
                for i in range(len(topk_indices)):
                    c = topk_indices[i].item()
                    prob_value = topk_probs[i].item()

                    # 非常に低い確率は無視（オプション）
                    if prob_value < -20:  # log_prob が非常に小さい値
                        continue

                    # 新しいスコアを計算
                    new_score = score + prob_value

                    # CTCの処理規則に基づいてビームを更新
                    if c == blank_id:
                        # blankの場合は、同じprefixでスコアを更新
                        new_beams.append((prefix, new_score, last_char, last_t))
                    elif (last_char == c) and (last_t == t - 1):
                        # 直前のタイムステップと同じ文字の場合（繰り返し）
                        # 1. 同じ文字を統合するパターン（CTCの繰り返し統合ルール）
                        new_beams.append((prefix, new_score, c, t))
                        # 2. 新しい文字として追加するパターン
                        new_beams.append((prefix + [c], new_score, c, t))
                    else:
                        # 新しい文字を追加
                        new_beams.append((prefix + [c], new_score, c, t))

            # 重複を除いてビームの幅を制限
            # ビームの幅制限前に重複を統合
            combined_beams = {}
            for prefix, score, last_char, last_t in new_beams:
                # タプルのリストを比較可能なようにtupleに変換
                prefix_tuple = tuple(prefix)
                key = (prefix_tuple, last_char, last_t)
                if key not in combined_beams or score > combined_beams[key][1]:
                    combined_beams[key] = (prefix, score, last_char, last_t)

            # スコアでソートしてビームの幅を制限
            sorted_beams = sorted(
                combined_beams.values(), key=lambda x: x[1], reverse=True
            )
            beams = sorted_beams[:beam_width]

            # 進捗状況を確認（最初のバッチのみ）
            if b == 0 and t % 10 == 0:
                print(
                    f"タイムステップ {t}/{seq_probs.size(0)}, 現在のトップビーム: {beams[0][0] if beams else []}"
                )

        # 最高スコアのビームを選択
        if beams:
            best_beam = max(beams, key=lambda x: x[1])[0]

            # 出力が全て同じトークンの場合は、セカンドベストの結果も検討
            if len(set(best_beam)) <= 1 and len(beams) > 1:
                print(
                    "警告: 最高スコアのビームがすべて同じトークンです。セカンドベストも確認します。"
                )
                second_best = beams[1][0] if len(beams) > 1 else []
                print(f"セカンドベスト: {second_best}")

                # もしセカンドベストがより多様であれば使用
                if len(set(second_best)) > len(set(best_beam)):
                    best_beam = second_best
        else:
            # ビームが空の場合（通常はあり得ないが念のため）
            best_beam = []

        # デバッグ用に出力
        if b == 0:  # 最初のバッチだけデバッグ情報を表示
            print(f"デコード結果サンプル: {best_beam}")

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