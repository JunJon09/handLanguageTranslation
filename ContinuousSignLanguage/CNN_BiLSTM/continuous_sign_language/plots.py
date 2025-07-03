import matplotlib.pyplot as plt
import numpy as np
import os
import CNN_BiLSTM.continuous_sign_language.config as config
from collections import Counter
import logging
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import umap
import pandas as pd
from matplotlib.colors import ListedColormap


def train_loss_plot(losses_default):

    plt.grid(axis="y", linestyle="dotted", color="k")

    xs = np.arange(1, len(losses_default) + 1)
    plt.plot(xs, losses_default, label="Default", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim([0.0, 5.0])
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(config.plot_save_dir, config.plot_loss_train_save_path)
    print(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def val_loss_plot(val_losses_default, eval_every_n_epochs):
    plt.grid(axis="y", linestyle="dotted", color="red")

    xs = np.arange(1, len(val_losses_default) * eval_every_n_epochs + 1)
    xs_val = np.arange(
        eval_every_n_epochs,
        len(val_losses_default) * eval_every_n_epochs + 1,
        eval_every_n_epochs,
    )
    plt.plot(xs_val, val_losses_default, label="Default", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim([0.0, 50.0])
    plt.xticks(np.arange(1, len(val_losses_default) + 1, eval_every_n_epochs))
    plt.legend()
    save_path = os.path.join(config.plot_save_dir, config.plot_loss_val_save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def train_val_loss_plot(train_losses_default, val_losses_default, eval_every_n_epochs):
    plt.grid(axis="y", linestyle="dotted", color="k")

    # Plot training loss
    num_epochs = len(train_losses_default)
    xs_train = np.arange(1, num_epochs + 1)
    plt.plot(xs_train, train_losses_default, label="Train", marker=".")

    # Plot validation loss with adjusted x-coordinates
    xs_val = np.arange(eval_every_n_epochs, num_epochs + 1, eval_every_n_epochs)
    plt.plot(xs_val, val_losses_default, label="Validation", marker=".")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim([0.0, 50.0])
    plt.legend()
    save_path = os.path.join(config.plot_save_dir, config.plot_loss_train_val_save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def test_data_plot(test_accs_default):
    plt.grid(axis="y", linestyle="dotted", color="k")

    xs = np.arange(1, len(test_accs_default) + 1)
    plt.plot(xs, test_accs_default, label="Default", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim([0.0, 100.0])
    plt.legend()
    save_path = os.path.join(config.plot_save_dir, config.plot_accuracy_save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def wer_plot(wer_scores_default, eval_every_n_epochs):
    plt.grid(axis="y", linestyle="dotted", color="red")

    xs = np.arange(1, len(wer_scores_default) * eval_every_n_epochs + 1)
    xs_val = np.arange(
        eval_every_n_epochs,
        len(wer_scores_default) * eval_every_n_epochs + 1,
        eval_every_n_epochs,
    )
    plt.plot(xs_val, wer_scores_default, label="Default", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("WER")
    plt.ylim([0.0, 1.5])
    plt.xticks(np.arange(1, len(wer_scores_default) + 1, eval_every_n_epochs))
    plt.legend()
    save_path = os.path.join(config.plot_save_dir, config.plot_wer_save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top5_trends(self, save_plot=True, plot_dir=None):
    """
    Top5トークンの頻度推移を折線グラフで表示して保存
    """
    if len(self.token_frequencies) == 0:
        print("分析データがありません")
        return

    # 全バッチの全トークンの累積頻度を計算
    all_tokens = Counter()
    for batch_list in self.token_frequencies:
        for batch_tokens in batch_list:
            for token, prob in batch_tokens.items():
                all_tokens[token] += prob

    # Top5トークンを特定
    top5_tokens = [token for token, _ in all_tokens.most_common(5)]

    # 各バッチでのTop5トークンの頻度推移を計算
    batch_numbers = list(range(len(self.token_frequencies)))
    token_trends = {token: [] for token in top5_tokens}

    for batch_list in self.token_frequencies:
        # 各バッチ内での平均頻度を計算
        batch_averages = {token: 0.0 for token in top5_tokens}

        for batch_tokens in batch_list:
            for token in top5_tokens:
                batch_averages[token] += batch_tokens.get(token, 0.0)

        # バッチサイズで平均化
        batch_size = len(batch_list) if batch_list else 1
        for token in top5_tokens:
            token_trends[token].append(batch_averages[token] / batch_size)

    # グラフの作成
    plt.figure(figsize=(12, 8))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]

    for i, token in enumerate(top5_tokens):
        plt.plot(
            batch_numbers,
            token_trends[token],
            label=f"{token}",
            color=colors[i],
            marker=markers[i],
            markersize=6,
            linewidth=2,
            alpha=0.8,
        )

    plt.xlabel("Batch Number", fontsize=12)
    plt.ylabel("Average Probability (Non-blank)", fontsize=12)
    plt.title("Top 5 Non-blank Token Trends", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 統計情報を表示
    print(
        f"\n=== Top 5 非ブランクトークン分析 (バッチ数: {len(self.token_frequencies)}) ==="
    )
    for i, token in enumerate(top5_tokens):
        final_prob = token_trends[token][-1] if token_trends[token] else 0
        max_prob = max(token_trends[token]) if token_trends[token] else 0
        avg_prob = np.mean(token_trends[token]) if token_trends[token] else 0
        print(f"{i+1}. {token}:")
        print(f"   最新: {final_prob:.4f}, 最大: {max_prob:.4f}, 平均: {avg_prob:.4f}")

    if save_plot:
        # 保存先ディレクトリの設定
        if plot_dir is None:
            plot_dir = config.plot_save_dir if hasattr(config, "plot_save_dir") else "."

        # ディレクトリが存在しない場合は作成
        os.makedirs(plot_dir, exist_ok=True)

        # ファイル名に日時を追加してユニークにする
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(plot_dir, f"non_blank_token_trends_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nグラフを保存しました: {save_path}")

    # plt.show() は削除
    plt.close()  # メモリリークを防ぐために明示的に閉じる


def plot_attention_matrix(
    attention_weights, sample_idx=0, save_path=None, title="Attention Matrix"
):
    """
    Attention重みをヒートマップとして可視化

    Args:
        attention_weights: numpy array of shape (batch_size, seq_len, seq_len)
        sample_idx: 可視化するバッチ内のサンプルインデックス
        save_path: 保存パス（Noneの場合は自動生成）
        title: グラフのタイトル
    """
    if attention_weights is None:
        logging.warning("Attention重みが取得できませんでした")
        return

    # バッチの最初のサンプルを取得
    if len(attention_weights.shape) == 3:
        attn_matrix = attention_weights[sample_idx]  # (seq_len, seq_len)
    else:
        attn_matrix = attention_weights

    plt.figure(figsize=(12, 10))

    # ヒートマップを作成
    im = plt.imshow(attn_matrix, cmap="Blues", aspect="auto", interpolation="nearest")

    plt.title(f"{title} (Sample {sample_idx})")
    plt.xlabel("Key Position (Time Steps)")
    plt.ylabel("Query Position (Time Steps)")

    # カラーバーを追加
    plt.colorbar(im, shrink=0.8)

    # 軸の設定
    seq_len = attn_matrix.shape[0]
    tick_positions = np.arange(0, seq_len, max(1, seq_len // 10))
    plt.xticks(tick_positions, tick_positions)
    plt.yticks(tick_positions, tick_positions)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(
            config.plot_save_dir, f"attention_matrix_sample_{sample_idx}.png"
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Attention matrix saved to: {save_path}")


def plot_attention_statistics(attention_weights, save_path=None):
    """
    Attention重みの統計情報を可視化

    Args:
        attention_weights: numpy array of shape (batch_size, seq_len, seq_len)
        save_path: 保存パス
    """
    if attention_weights is None:
        logging.warning("Attention重みが取得できませんでした")
        return

    # バッチの最初のサンプルを使用
    if len(attention_weights.shape) == 3:
        attn_matrix = attention_weights[0]
    else:
        attn_matrix = attention_weights

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Attention重みの分布
    axes[0, 0].hist(attn_matrix.flatten(), bins=50, alpha=0.7, color="blue")
    axes[0, 0].set_title("Distribution of Attention Weights")
    axes[0, 0].set_xlabel("Attention Weight")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 各クエリ位置での最大注意重み
    max_attention_per_query = np.max(attn_matrix, axis=1)
    axes[0, 1].plot(max_attention_per_query, marker="o", markersize=3)
    axes[0, 1].set_title("Maximum Attention Weight per Query Position")
    axes[0, 1].set_xlabel("Query Position")
    axes[0, 1].set_ylabel("Max Attention Weight")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 対角線要素（自己注意の強さ）
    diagonal_attention = np.diag(attn_matrix)
    axes[1, 0].plot(diagonal_attention, marker="o", markersize=3, color="red")
    axes[1, 0].set_title("Self-Attention Strength (Diagonal Elements)")
    axes[1, 0].set_xlabel("Position")
    axes[1, 0].set_ylabel("Self-Attention Weight")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 注意の集中度（エントロピー）
    epsilon = 1e-8  # ゼロ除算を防ぐ
    entropy_per_query = -np.sum(attn_matrix * np.log(attn_matrix + epsilon), axis=1)
    axes[1, 1].plot(entropy_per_query, marker="o", markersize=3, color="green")
    axes[1, 1].set_title("Attention Entropy per Query Position")
    axes[1, 1].set_xlabel("Query Position")
    axes[1, 1].set_ylabel("Entropy (lower = more focused)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.plot_save_dir, "attention_statistics.png")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Attention statistics saved to: {save_path}")


def plot_attention_focus_over_time(attention_weights, save_path=None, window_size=10):
    """
    時間経過に伴うAttentionの焦点の変化を可視化

    Args:
        attention_weights: numpy array of shape (batch_size, seq_len, seq_len)
        save_path: 保存パス
        window_size: 移動窓のサイズ
    """
    if attention_weights is None:
        logging.warning("Attention重みが取得できませんでした")
        return

    # バッチの最初のサンプルを使用
    if len(attention_weights.shape) == 3:
        attn_matrix = attention_weights[0]
    else:
        attn_matrix = attention_weights

    seq_len = attn_matrix.shape[0]

    plt.figure(figsize=(15, 8))

    # 各クエリ位置で最も注目しているキー位置を計算
    max_attention_positions = np.argmax(attn_matrix, axis=1)

    plt.subplot(2, 1, 1)
    plt.plot(max_attention_positions, marker="o", markersize=3, linewidth=2)
    plt.title("Most Attended Position over Time")
    plt.xlabel("Query Position (Time)")
    plt.ylabel("Most Attended Key Position")
    plt.grid(True, alpha=0.3)

    # 対角線を追加（自己注意の参考）
    plt.plot(
        [0, seq_len - 1],
        [0, seq_len - 1],
        "r--",
        alpha=0.5,
        label="Self-attention line",
    )
    plt.legend()

    # 移動平均を計算して平滑化
    if seq_len > window_size:
        moving_avg = np.convolve(
            max_attention_positions, np.ones(window_size) / window_size, mode="valid"
        )
        # moving_avgの長さに合わせてx座標を生成
        x_smooth = np.arange(window_size // 2, window_size // 2 + len(moving_avg))

        # デバッグ情報をログに記録
        logging.debug(f"seq_len: {seq_len}, window_size: {window_size}")
        logging.debug(f"max_attention_positions.shape: {max_attention_positions.shape}")
        logging.debug(
            f"moving_avg.shape: {moving_avg.shape}, x_smooth.shape: {x_smooth.shape}"
        )

        # 長さが一致していることを確認
        if len(x_smooth) != len(moving_avg):
            logging.error(
                f"x_smooth length ({len(x_smooth)}) != moving_avg length ({len(moving_avg)})"
            )
            return

        plt.subplot(2, 1, 2)
        plt.plot(max_attention_positions, alpha=0.3, label="Original")
        plt.plot(
            x_smooth,
            moving_avg,
            linewidth=2,
            label=f"Moving Average (window={window_size})",
        )
        plt.title("Smoothed Attention Focus Trajectory")
        plt.xlabel("Query Position (Time)")
        plt.ylabel("Most Attended Key Position")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.plot_save_dir, "attention_focus_over_time.png")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Attention focus over time saved to: {save_path}")


def plot_ctc_probability_heatmap(
    log_probs, vocab_dict=None, blank_id=0, save_path=None, sample_idx=0
):
    """
    CTC確率のヒートマップを可視化

    Args:
        log_probs: torch.Tensor of shape (T, B, C) - CTC出力の対数確率
        vocab_dict: 語彙辞書 (optional)
        blank_id: ブランクトークンのID
        save_path: 保存パス
        sample_idx: 可視化するバッチのサンプルインデックス
    """
    try:
        if log_probs is None:
            logging.warning("CTC log_probsが取得できませんでした")
            return False

        # log_probsの形状を確認・調整
        if len(log_probs.shape) == 3:
            # (T, B, C) -> (B, T, C)
            if log_probs.size(1) == 1:  # バッチサイズが1の場合
                probs = torch.exp(log_probs[:, sample_idx, :]).cpu().numpy()  # (T, C)
            else:
                probs = (
                    torch.exp(log_probs).permute(1, 0, 2)[sample_idx].cpu().numpy()
                )  # (T, C)
        else:
            probs = torch.exp(log_probs).cpu().numpy()

        time_steps, num_classes = probs.shape

        # 表示するクラス数を制限（上位10クラス + ブランク）
        mean_probs = probs.mean(axis=0)
        top_n = min(10, num_classes - 1)
        top_classes = np.argsort(mean_probs)[-top_n:]

        # ブランクIDを含める
        display_classes = list(top_classes)
        if blank_id not in display_classes:
            display_classes.append(blank_id)
        display_classes.sort()

        # 選択したクラスのみのヒートマップを表示
        selected_probs = probs[:, display_classes]

        plt.figure(figsize=(14, 8))

        # ヒートマップ
        im = plt.imshow(
            selected_probs.T, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        plt.colorbar(im, label="確率")

        # 軸ラベル
        plt.xlabel("タイムステップ (フレーム)", fontsize=12)
        plt.ylabel("クラス", fontsize=12)
        plt.title(
            f"CTC確率ヒートマップ (Sample {sample_idx})", fontsize=14, fontweight="bold"
        )

        # Y軸のクラスID表示
        class_labels = []
        for idx in display_classes:
            if vocab_dict and idx in vocab_dict:
                label = f"{idx}:{vocab_dict[idx][:8]}"  # 語彙名を8文字で制限
            else:
                label = f"{idx}"
            if idx == blank_id:
                label += " (Blank)"
            class_labels.append(label)

        plt.yticks(range(len(display_classes)), class_labels, fontsize=10)

        # X軸の設定
        tick_positions = np.arange(0, time_steps, max(1, time_steps // 10))
        plt.xticks(tick_positions, tick_positions)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(
                config.plot_save_dir, f"ctc_heatmap_sample_{sample_idx}.png"
            )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"CTC確率ヒートマップを保存: {save_path}")

        return True

    except Exception as e:
        logging.error(f"CTC確率ヒートマップ生成でエラー: {e}")
        return False


def plot_ctc_probability_over_time(
    log_probs, vocab_dict=None, blank_id=0, save_path=None, sample_idx=0, top_k=5
):
    """
    フレームごとのCTC確率変化を可視化

    Args:
        log_probs: torch.Tensor of shape (T, B, C) - CTC出力の対数確率
        vocab_dict: 語彙辞書 (optional)
        blank_id: ブランクトークンのID
        save_path: 保存パス
        sample_idx: 可視化するバッチのサンプルインデックス
        top_k: 表示する上位クラス数
    """
    try:

        if log_probs is None:
            logging.warning("CTC log_probsが取得できませんでした")
            return False

        # log_probsの形状を確認・調整
        if len(log_probs.shape) == 3:
            if log_probs.size(1) == 1:  # バッチサイズが1の場合
                probs = torch.exp(log_probs[:, sample_idx, :]).cpu().numpy()  # (T, C)
            else:
                probs = (
                    torch.exp(log_probs).permute(1, 0, 2)[sample_idx].cpu().numpy()
                )  # (T, C)
        else:
            probs = torch.exp(log_probs).cpu().numpy()

        time_steps, num_classes = probs.shape

        # 全体の平均確率でTop-kクラスを決定
        mean_probs = probs.mean(axis=0)
        top_k_indices = np.argsort(mean_probs)[-top_k:]

        plt.figure(figsize=(12, 8))

        # カラーパレット
        colors = plt.cm.tab10(np.linspace(0, 1, top_k + 1))

        # 上位k個のクラスの時間変化をプロット
        for i, class_idx in enumerate(top_k_indices):
            class_probs_over_time = probs[:, class_idx]

            # クラス名の取得
            if vocab_dict and class_idx in vocab_dict:
                label = f"Class {class_idx}: {vocab_dict[class_idx][:10]}"
            else:
                label = f"Class {class_idx}"

            plt.plot(
                range(time_steps),
                class_probs_over_time,
                label=label,
                linewidth=2,
                color=colors[i],
                alpha=0.8,
            )

        # ブランクIDを特別に表示（Top-kに含まれていない場合）
        if blank_id not in top_k_indices:
            blank_probs_over_time = probs[:, blank_id]
            blank_label = "Blank"
            if vocab_dict and blank_id in vocab_dict:
                blank_label = f"Blank: {vocab_dict[blank_id]}"

            plt.plot(
                range(time_steps),
                blank_probs_over_time,
                label=blank_label,
                linewidth=2,
                linestyle="--",
                color="black",
                alpha=0.9,
            )

        # グラフの設定
        plt.xlabel("タイムステップ (フレーム)", fontsize=12)
        plt.ylabel("確率", fontsize=12)
        plt.title(
            f"フレームごとのCTC確率変化 (Sample {sample_idx})",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        # 確率の平均値を表示
        top_k_mean_probs = mean_probs[top_k_indices]
        stats_text = f"平均確率 Top{top_k}: {[f'{p:.4f}' for p in top_k_mean_probs]}"
        plt.figtext(
            0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor="white", alpha=0.8)
        )

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(
                config.plot_save_dir, f"ctc_prob_over_time_sample_{sample_idx}.png"
            )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"CTC確率時間変化を保存: {save_path}")

        return True

    except Exception as e:
        logging.error(f"CTC確率時間変化生成でエラー: {e}")
        return False


def plot_ctc_statistics(log_probs, blank_id=0, save_path=None, sample_idx=0):
    """
    CTC確率の統計情報を可視化

    Args:
        log_probs: torch.Tensor of shape (T, B, C) - CTC出力の対数確率
        blank_id: ブランクトークンのID
        save_path: 保存パス
        sample_idx: 可視化するバッチのサンプルインデックス
    """
    try:

        if log_probs is None:
            logging.warning("CTC log_probsが取得できませんでした")
            return False

        # log_probsの形状を確認・調整
        if len(log_probs.shape) == 3:
            if log_probs.size(1) == 1:  # バッチサイズが1の場合
                probs = torch.exp(log_probs[:, sample_idx, :]).cpu().numpy()  # (T, C)
            else:
                probs = (
                    torch.exp(log_probs).permute(1, 0, 2)[sample_idx].cpu().numpy()
                )  # (T, C)
        else:
            probs = torch.exp(log_probs).cpu().numpy()

        time_steps, num_classes = probs.shape

        # フレームごとの最大確率クラスを計算
        max_probs = np.max(probs, axis=1)
        max_indices = np.argmax(probs, axis=1)

        # 各クラスの統計
        unique_classes, counts = np.unique(max_indices, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]  # 降順ソート

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 確率分布のヒストグラム
        axes[0, 0].hist(
            probs.flatten(), bins=50, alpha=0.7, color="blue", edgecolor="black"
        )
        axes[0, 0].set_title("CTC確率の分布", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("確率")
        axes[0, 0].set_ylabel("頻度")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. フレームごとの最大確率
        axes[0, 1].plot(max_probs, marker="o", markersize=2, linewidth=1)
        axes[0, 1].set_title("フレームごとの最大確率", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("フレーム")
        axes[0, 1].set_ylabel("最大確率")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 最頻出クラス統計（上位10クラス）
        top_10_classes = unique_classes[sorted_idx[:10]]
        top_10_counts = counts[sorted_idx[:10]]
        top_10_percentages = top_10_counts / time_steps * 100

        bars = axes[1, 0].bar(
            range(len(top_10_classes)),
            top_10_percentages,
            color=["red" if cls == blank_id else "blue" for cls in top_10_classes],
        )
        axes[1, 0].set_title(
            "最頻出クラス統計 (上位10)", fontsize=12, fontweight="bold"
        )
        axes[1, 0].set_xlabel("クラスID")
        axes[1, 0].set_ylabel("出現率 (%)")
        axes[1, 0].set_xticks(range(len(top_10_classes)))
        axes[1, 0].set_xticklabels(
            [f"{cls}" + ("(B)" if cls == blank_id else "") for cls in top_10_classes],
            rotation=45,
        )
        axes[1, 0].grid(True, alpha=0.3)

        # ブランクトークンのバーを強調
        for i, cls in enumerate(top_10_classes):
            if cls == blank_id:
                bars[i].set_color("red")
                bars[i].set_alpha(0.8)

        # 4. ブランクトークンと非ブランクトークンの比較
        blank_frames = np.sum(max_indices == blank_id)
        non_blank_frames = time_steps - blank_frames

        labels = ["Non-blank", "Blank"]
        sizes = [non_blank_frames, blank_frames]
        colors = ["lightblue", "lightcoral"]

        wedges, texts, autotexts = axes[1, 1].pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
        )
        axes[1, 1].set_title("ブランク vs 非ブランク", fontsize=12, fontweight="bold")

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(
                config.plot_save_dir, f"ctc_statistics_sample_{sample_idx}.png"
            )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 統計情報をログに出力
        logging.info(f"CTC統計情報 (Sample {sample_idx}):")
        logging.info(f"  総フレーム数: {time_steps}")
        logging.info(
            f"  ブランクフレーム: {blank_frames} ({blank_frames/time_steps*100:.1f}%)"
        )
        logging.info(
            f"  非ブランクフレーム: {non_blank_frames} ({non_blank_frames/time_steps*100:.1f}%)"
        )
        logging.info(f"  最大確率の平均: {np.mean(max_probs):.4f}")
        logging.info(f"CTC統計グラフを保存: {save_path}")

        return True

    except Exception as e:
        logging.error(f"CTC統計情報生成でエラー: {e}")
        return False


def visualize_ctc_alignment_path(
    log_probs,
    decoded_sequence=None,
    target_sequence=None,
    vocab_dict=None,
    blank_id=0,
    sample_idx=0,
    save_dir=None,
):
    """
    CTC Alignment Pathの包括的な可視化

    Args:
        log_probs: torch.Tensor of shape (T, B, C) - CTC出力の対数確率
        decoded_sequence: デコード結果のシーケンス
        target_sequence: 正解シーケンス (optional)
        vocab_dict: 語彙辞書 (optional)
        blank_id: ブランクトークンのID
        sample_idx: 可視化するバッチのサンプルインデックス
        save_dir: 保存ディレクトリ
    """
    if log_probs is None:
        logging.warning("CTC log_probsが取得できませんでした")
        return False

    if save_dir is None:
        save_dir = config.plot_save_dir

    os.makedirs(save_dir, exist_ok=True)

    success_count = 0
    total_plots = 3

    # 1. 確率ヒートマップ
    heatmap_path = os.path.join(save_dir, f"ctc_heatmap_sample_{sample_idx}.png")
    logging.info(f"CTC ヒートマップ生成開始: {heatmap_path}")
    if plot_ctc_probability_heatmap(
        log_probs, vocab_dict, blank_id, heatmap_path, sample_idx
    ):
        success_count += 1
        logging.info(f"CTC ヒートマップ生成成功: {heatmap_path}")
    else:
        logging.warning(f"CTC ヒートマップ生成失敗: {heatmap_path}")

    # 2. 時間変化グラフ
    time_path = os.path.join(save_dir, f"ctc_prob_time_sample_{sample_idx}.png")
    logging.info(f"CTC 時間変化グラフ生成開始: {time_path}")
    if plot_ctc_probability_over_time(
        log_probs, vocab_dict, blank_id, time_path, sample_idx
    ):
        success_count += 1
        logging.info(f"CTC 時間変化グラフ生成成功: {time_path}")
    else:
        logging.warning(f"CTC 時間変化グラフ生成失敗: {time_path}")

    # 3. 統計情報
    stats_path = os.path.join(save_dir, f"ctc_statistics_sample_{sample_idx}.png")
    if plot_ctc_statistics(log_probs, blank_id, stats_path, sample_idx):
        success_count += 1

    # 結果のログ出力
    if decoded_sequence is not None:
        decoded_text = " ".join([str(token) for token in decoded_sequence])
        logging.info(f"デコード結果: {decoded_text}")

    if target_sequence is not None:
        target_text = " ".join([str(token) for token in target_sequence])
        logging.info(f"正解シーケンス: {target_text}")

    logging.info(f"CTC可視化完了: {success_count}/{total_plots} 個のグラフを生成")

    return success_count == total_plots


# === 新しい分析・可視化機能 ===


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    save_path=None,
    normalize=True,
    title="Confusion Matrix",
    figsize=(12, 10),
):
    """
    混同行列を可視化

    Args:
        y_true: 正解ラベルのリスト/配列
        y_pred: 予測ラベルのリスト/配列
        class_names: クラス名のリスト (optional)
        save_path: 保存パス
        normalize: 正規化するかどうか
        title: グラフのタイトル
        figsize: 図のサイズ
    """
    try:
        # 必要なライブラリをここでインポート
        try:
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
            import pandas as pd
        except ImportError as e:
            logging.error(f"必要なライブラリがインストールされていません: {e}")
            logging.info("以下のコマンドでインストールしてください:")
            logging.info("pip install seaborn scikit-learn pandas")
            return False

        # 混同行列を計算
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            # 正規化 (行ごとに正規化 - 各クラスの正解数で割る)
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # NaNを0に変換
        else:
            cm_normalized = cm

        # 図を作成
        plt.figure(figsize=figsize)

        # クラス名の設定
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]

        # ヒートマップを描画
        if normalize:
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".3f",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={"label": "Normalized Frequency"},
            )
            plt.title(f"{title} (Normalized)")
        else:
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={"label": "Count"},
            )
            plt.title(f"{title} (Raw Counts)")

        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # 統計情報を追加
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(
            0.02,
            0.02,
            f"Overall Accuracy: {accuracy:.3f}",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(config.plot_save_dir, "confusion_matrix.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"混同行列を保存: {save_path}")

        # 詳細な分析結果をログに出力
        log_confusion_analysis(cm, class_names, accuracy)

        return True

    except Exception as e:
        logging.error(f"混同行列生成でエラー: {e}")
        return False


def log_confusion_analysis(cm, class_names, overall_accuracy):
    """
    混同行列の詳細分析をログに出力

    Args:
        cm: 混同行列
        class_names: クラス名
        overall_accuracy: 全体精度
    """
    try:
        logging.info("=== 混同行列分析結果 ===")
        logging.info(f"全体精度: {overall_accuracy:.3f}")

        # クラスごとの精度・再現率・F1スコアを計算
        num_classes = len(cm)

        for i in range(num_classes):
            # True Positive, False Positive, False Negative
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            # 精度 (Precision)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            # 再現率 (Recall)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # F1スコア
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
            logging.info(
                f"{class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}"
            )

        # 最も間違えやすいペアを特定
        logging.info("\n=== 主な誤分類パターン ===")
        confusion_pairs = []

        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and cm[i, j] > 0:
                    true_class = (
                        class_names[i] if i < len(class_names) else f"Class_{i}"
                    )
                    pred_class = (
                        class_names[j] if j < len(class_names) else f"Class_{j}"
                    )
                    confusion_pairs.append((cm[i, j], true_class, pred_class))

        # 間違いの多い順にソート
        confusion_pairs.sort(reverse=True)

        # 上位5つの誤分類パターンを表示
        for count, true_cls, pred_cls in confusion_pairs[:5]:
            logging.info(f"{true_cls} -> {pred_cls}: {count}回")

    except Exception as e:
        logging.error(f"混同行列分析でエラー: {e}")


def analyze_word_level_confusion(
    predictions, ground_truth, vocab_dict=None, save_path=None
):
    """
    単語レベルでの混同行列分析

    Args:
        predictions: 予測結果のリスト (単語IDまたは単語文字列)
        ground_truth: 正解ラベルのリスト
        vocab_dict: 語彙辞書 (ID -> 単語名)
        save_path: 保存パス
    """
    try:
        logging.info("単語レベル混同行列分析を開始")

        # 語彙辞書から単語名を取得
        if vocab_dict:
            unique_labels = sorted(set(ground_truth + predictions))
            class_names = []
            for label in unique_labels:
                if label in vocab_dict:
                    # 単語名を短縮表示 (長すぎる場合)
                    word_name = vocab_dict[label]
                    if len(word_name) > 10:
                        word_name = word_name[:8] + ".."
                    class_names.append(f"{label}:{word_name}")
                else:
                    class_names.append(f"ID_{label}")
        else:
            class_names = None

        # 混同行列を生成
        success = plot_confusion_matrix(
            y_true=ground_truth,
            y_pred=predictions,
            class_names=class_names,
            save_path=save_path,
            normalize=True,
            title="Word-Level Confusion Matrix",
            figsize=(15, 12),
        )

        if success:
            logging.info("単語レベル混同行列分析が完了")

        return success

    except Exception as e:
        logging.error(f"単語レベル混同行列分析でエラー: {e}")
        return False


# === 時系列予測確率・信頼度可視化機能 ===


def plot_prediction_confidence_over_time(
    log_probs, vocab_dict=None, save_path=None, sample_idx=0
):
    """
    予測信頼度の時系列変化を可視化

    Args:
        log_probs: torch.Tensor of shape (T, B, C) - CTC出力の対数確率
        vocab_dict: 語彙辞書 (optional)
        save_path: 保存パス
        sample_idx: 可視化するバッチのサンプルインデックス
    """
    try:
        if log_probs is None:
            logging.warning("CTC log_probsが取得できませんでした")
            return False

        # torch.Tensorかどうかチェック
        if not isinstance(log_probs, torch.Tensor):
            logging.warning(f"log_probsの型が不正です: {type(log_probs)}")
            return False

        # NaNや無限大値をチェック
        if torch.isnan(log_probs).any():
            logging.warning("log_probsにNaN値が含まれています")
            return False
        
        if torch.isinf(log_probs).any():
            logging.warning("log_probsに無限大値が含まれています")
            return False

        # log_probsの形状を確認・調整
        if len(log_probs.shape) == 3:
            if sample_idx >= log_probs.size(1):
                logging.warning(f"sample_idx {sample_idx} がバッチサイズを超えています: {log_probs.size(1)}")
                sample_idx = 0
            if log_probs.size(1) == 1:  # バッチサイズが1の場合
                probs = torch.exp(log_probs[:, sample_idx, :]).cpu().numpy()  # (T, C)
            else:
                probs = (
                    torch.exp(log_probs).permute(1, 0, 2)[sample_idx].cpu().numpy()
                )  # (T, C)
        else:
            probs = torch.exp(log_probs).cpu().numpy()

        time_steps, num_classes = probs.shape

        logging.info(f"信頼度可視化データ準備完了: {probs.shape}")

        # 信頼度指標を計算
        max_probs = np.max(probs, axis=1)  # 各時刻での最大確率

        # エントロピー計算（予測の不確実性）
        epsilon = 1e-8
        entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)

        # 信頼度スコア（最大確率と2番目の確率の差）
        sorted_probs = np.sort(probs, axis=1)
        confidence_gap = sorted_probs[:, -1] - sorted_probs[:, -2]  # 1位と2位の差

        # 確実性指数（最大確率の相対的強さ）
        prob_std = np.std(probs, axis=1)  # 各時刻での確率分布の標準偏差
        certainty_index = max_probs / (prob_std + epsilon)

        # 4つのサブプロットで可視化
        plt.figure(figsize=(16, 12))

        # 1. 最大確率の時系列変化
        plt.subplot(2, 2, 1)
        plt.plot(
            range(time_steps), max_probs, "b-", linewidth=2, marker="o", markersize=3
        )
        plt.title(
            "Maximum Prediction Probability Over Time", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Time Step (Frame)", fontsize=12)
        plt.ylabel("Max Probability", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        # 平均値ラインを追加
        mean_max_prob = np.mean(max_probs)
        plt.axhline(
            y=mean_max_prob,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {mean_max_prob:.3f}",
        )
        plt.legend()

        # 2. 予測エントロピー（不確実性）
        plt.subplot(2, 2, 2)
        plt.plot(
            range(time_steps), entropy, "g-", linewidth=2, marker="s", markersize=3
        )
        plt.title(
            "Prediction Entropy (Uncertainty) Over Time", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Time Step (Frame)", fontsize=12)
        plt.ylabel("Entropy (higher = more uncertain)", fontsize=12)
        plt.grid(True, alpha=0.3)

        # 平均値と閾値ライン
        mean_entropy = np.mean(entropy)
        plt.axhline(
            y=mean_entropy,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {mean_entropy:.3f}",
        )

        # 高不確実性の閾値（上位25%）
        high_uncertainty_threshold = np.percentile(entropy, 75)
        plt.axhline(
            y=high_uncertainty_threshold,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label=f"High Uncertainty: {high_uncertainty_threshold:.3f}",
        )
        plt.legend()

        # 3. 信頼度ギャップ（1位と2位の差）
        plt.subplot(2, 2, 3)
        plt.plot(
            range(time_steps),
            confidence_gap,
            "purple",
            linewidth=2,
            marker="^",
            markersize=3,
        )
        plt.title(
            "Confidence Gap (1st - 2nd Probability)", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Time Step (Frame)", fontsize=12)
        plt.ylabel("Probability Gap", fontsize=12)
        plt.grid(True, alpha=0.3)

        mean_gap = np.mean(confidence_gap)
        plt.axhline(
            y=mean_gap,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {mean_gap:.3f}",
        )

        # 低信頼度の閾値（下位25%）
        low_confidence_threshold = np.percentile(confidence_gap, 25)
        plt.axhline(
            y=low_confidence_threshold,
            color="red",
            linestyle=":",
            alpha=0.7,
            label=f"Low Confidence: {low_confidence_threshold:.3f}",
        )
        plt.legend()

        # 4. 確実性指数
        plt.subplot(2, 2, 4)
        plt.plot(
            range(time_steps),
            certainty_index,
            "orange",
            linewidth=2,
            marker="d",
            markersize=3,
        )
        plt.title(
            "Certainty Index (Max Prob / Std Dev)", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Time Step (Frame)", fontsize=12)
        plt.ylabel("Certainty Index", fontsize=12)
        plt.grid(True, alpha=0.3)

        mean_certainty = np.mean(certainty_index)
        plt.axhline(
            y=mean_certainty,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {mean_certainty:.3f}",
        )
        plt.legend()

        plt.tight_layout()

        # 統計情報をテキストで追加
        stats_text = f"""Confidence Statistics:
Max Prob: μ={mean_max_prob:.3f}, σ={np.std(max_probs):.3f}
Entropy: μ={mean_entropy:.3f}, σ={np.std(entropy):.3f}
Confidence Gap: μ={mean_gap:.3f}, σ={np.std(confidence_gap):.3f}
Certainty Index: μ={mean_certainty:.3f}, σ={np.std(certainty_index):.3f}"""

        plt.figtext(
            0.02,
            0.02,
            stats_text,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
        )

        if save_path is None:
            save_path = os.path.join(
                config.plot_save_dir,
                f"prediction_confidence_timeline_sample_{sample_idx}.png",
            )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"予測信頼度タイムラインを保存: {save_path}")

        # 詳細な分析結果をログに出力
        log_confidence_analysis(max_probs, entropy, confidence_gap, certainty_index)

        return True

    except Exception as e:
        logging.error(f"予測信頼度可視化でエラー: {e}")
        import traceback
        logging.error(f"詳細なエラー情報: {traceback.format_exc()}")
        return False


def log_confidence_analysis(max_probs, entropy, confidence_gap, certainty_index):
    """
    信頼度分析の詳細結果をログに出力

    Args:
        max_probs: 最大確率の配列
        entropy: エントロピーの配列
        confidence_gap: 信頼度ギャップの配列
        certainty_index: 確実性指数の配列
    """
    try:
        logging.info("=== 予測信頼度分析結果 ===")

        # 基本統計
        logging.info(
            f"最大確率 - 平均: {np.mean(max_probs):.4f}, 標準偏差: {np.std(max_probs):.4f}"
        )
        logging.info(
            f"エントロピー - 平均: {np.mean(entropy):.4f}, 標準偏差: {np.std(entropy):.4f}"
        )
        logging.info(
            f"信頼度ギャップ - 平均: {np.mean(confidence_gap):.4f}, 標準偏差: {np.std(confidence_gap):.4f}"
        )
        logging.info(
            f"確実性指数 - 平均: {np.mean(certainty_index):.4f}, 標準偏差: {np.std(certainty_index):.4f}"
        )

        # 高信頼度・低信頼度フレームの特定
        high_confidence_frames = np.where(max_probs > np.percentile(max_probs, 75))[0]
        low_confidence_frames = np.where(max_probs < np.percentile(max_probs, 25))[0]

        logging.info(f"\n高信頼度フレーム (上位25%): {len(high_confidence_frames)}個")
        if len(high_confidence_frames) > 0:
            logging.info(
                f"フレーム範囲: {high_confidence_frames[0]} - {high_confidence_frames[-1]}"
            )

        logging.info(f"低信頼度フレーム (下位25%): {len(low_confidence_frames)}個")
        if len(low_confidence_frames) > 0:
            logging.info(
                f"フレーム範囲: {low_confidence_frames[0]} - {low_confidence_frames[-1]}"
            )

        # 不確実性が高い時点を特定
        high_uncertainty_frames = np.where(entropy > np.percentile(entropy, 75))[0]
        logging.info(
            f"\n高不確実性フレーム (エントロピー上位25%): {len(high_uncertainty_frames)}個"
        )

        # 予測の安定性評価
        prob_volatility = np.std(np.diff(max_probs))  # 最大確率の変動性
        entropy_volatility = np.std(np.diff(entropy))  # エントロピーの変動性

        logging.info(f"\n予測安定性:")
        logging.info(f"確率変動性: {prob_volatility:.4f}")
        logging.info(f"不確実性変動性: {entropy_volatility:.4f}")

        # 総合評価
        overall_confidence = np.mean(max_probs) * (
            1 - np.mean(entropy) / np.log(len(entropy))
        )
        logging.info(f"\n総合信頼度スコア: {overall_confidence:.4f}")

    except Exception as e:
        logging.error(f"信頼度分析ログ出力でエラー: {e}")


def plot_word_level_confidence_timeline(
    predictions,
    log_probs,
    word_timestamps=None,
    vocab_dict=None,
    save_path=None,
    sample_idx=0,
):
    """
    単語レベルでの予測信頼度タイムラインを可視化

    Args:
        predictions: 予測された単語のリスト
        log_probs: CTC出力の対数確率
        word_timestamps: 各単語のタイムスタンプ (optional)
        vocab_dict: 語彙辞書 (optional)
        save_path: 保存パス
        sample_idx: サンプルインデックス
    """
    try:
        # 入力データの妥当性チェック
        if log_probs is None:
            logging.warning("log_probsがNullです")
            return False
        
        if not predictions or len(predictions) == 0:
            logging.warning("予測結果が空です")
            return False

        # torch.Tensorかどうかチェック
        if not isinstance(log_probs, torch.Tensor):
            logging.warning(f"log_probsの型が不正です: {type(log_probs)}")
            return False

        # NaNや無限大値をチェック
        if torch.isnan(log_probs).any():
            logging.warning("log_probsにNaN値が含まれています")
            return False
        
        if torch.isinf(log_probs).any():
            logging.warning("log_probsに無限大値が含まれています")
            return False

        # 確率データの準備
        if len(log_probs.shape) == 3:
            if sample_idx >= log_probs.size(1):
                logging.warning(f"sample_idx {sample_idx} がバッチサイズを超えています: {log_probs.size(1)}")
                sample_idx = 0
            probs = torch.exp(log_probs[:, sample_idx, :]).cpu().numpy()
        else:
            probs = torch.exp(log_probs).cpu().numpy()

        time_steps = probs.shape[0]
        vocab_size = probs.shape[1]

        logging.info(f"確率データ準備完了: {probs.shape}, 予測単語数: {len(predictions)}")

        # 単語ごとの信頼度計算
        word_confidences = []
        word_names = []

        for i, word_id in enumerate(predictions):
            try:
                if isinstance(word_id, str):
                    word_name = word_id
                    # 語彙辞書から対応するIDを検索
                    word_confidence = 0.0  # デフォルト値
                else:
                    word_name = (
                        vocab_dict.get(word_id, f"Word_{word_id}")
                        if vocab_dict
                        else f"Word_{word_id}"
                    )
                    # 該当単語の平均確率を計算
                    if word_id < vocab_size:
                        word_confidence = np.mean(probs[:, word_id])
                    else:
                        logging.warning(f"word_id {word_id} が語彙サイズ {vocab_size} を超えています")
                        word_confidence = 0.0

                word_confidences.append(word_confidence)
                word_names.append(word_name)
            except Exception as e:
                logging.warning(f"単語 {i} の信頼度計算でエラー: {e}")
                word_confidences.append(0.0)
                word_names.append(f"Error_{i}")

        if not word_confidences:
            logging.warning("単語信頼度が計算できませんでした")
            return False

        # 可視化
        plt.figure(figsize=(14, 8))

        # 単語別信頼度バープロット
        plt.subplot(2, 1, 1)
        bars = plt.bar(
            range(len(word_names)),
            word_confidences,
            color=plt.cm.viridis(np.array(word_confidences)),
        )
        plt.title("Word-Level Prediction Confidence", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Words", fontsize=12)
        plt.ylabel("Average Confidence", fontsize=12)
        plt.xticks(range(len(word_names)), word_names, rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")

        # 信頼度閾値ライン
        confidence_threshold = 0.5
        plt.axhline(
            y=confidence_threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Confidence Threshold: {confidence_threshold}",
        )
        plt.legend()

        # 各バーに数値を表示
        for bar, conf in zip(bars, word_confidences):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{conf:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # 時系列での最大確率変化
        plt.subplot(2, 1, 2)
        max_probs_over_time = np.max(probs, axis=1)
        plt.plot(range(time_steps), max_probs_over_time, "b-", linewidth=2, alpha=0.7)
        plt.fill_between(range(time_steps), max_probs_over_time, alpha=0.3)
        plt.title(
            "Maximum Prediction Probability Timeline", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Time Step (Frame)", fontsize=12)
        plt.ylabel("Max Probability", fontsize=12)
        plt.grid(True, alpha=0.3)

        # 単語境界を表示（タイムスタンプがある場合）
        if word_timestamps:
            for i, timestamp in enumerate(word_timestamps):
                if i < len(word_names):
                    plt.axvline(x=timestamp, color="red", linestyle=":", alpha=0.6)
                    plt.text(
                        timestamp,
                        0.9,
                        word_names[i],
                        rotation=90,
                        verticalalignment="bottom",
                        fontsize=8,
                    )

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(
                config.plot_save_dir,
                f"word_confidence_timeline_sample_{sample_idx}.png",
            )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"単語レベル信頼度タイムラインを保存: {save_path}")

        # 単語別信頼度をログ出力
        logging.info("=== 単語別予測信頼度 ===")
        for word, conf in zip(word_names, word_confidences):
            confidence_level = "高" if conf > 0.7 else "中" if conf > 0.4 else "低"
            logging.info(f"{word}: {conf:.4f} ({confidence_level}信頼度)")

        return True

    except Exception as e:
        logging.error(f"単語レベル信頼度可視化でエラー: {e}")
        import traceback
        logging.error(f"詳細なエラー情報: {traceback.format_exc()}")
        return False


def extract_multilayer_features(
    model,
    feature,
    spatial_feature,
    tokens,
    feature_pad_mask,
    input_lengths,
    target_lengths,
    blank_id,
):
    """
    手話認識モデルの複数層から特徴量を抽出
    
    各層の意味:
    - CNN出力: 空間的パターン（手の形、位置関係、顔の表情、体の姿勢）
    - BiLSTM隠れ状態: 時系列ダイナミクス（動きの軌跡、速度、手話の文脈情報）
    - Attention重み: 重要度マップ（どの時刻・部位に注目しているか、手話認識の判断根拠）
    - 最終層直前: 統合的判断（全情報を統合した最終特徴量、分類に直結する表現）

    Args:
        model: CNNBiLSTMモデル
        feature: 入力特徴量 (B, C, T, J)
        spatial_feature: 空間特徴量
        tokens: ターゲットトークン
        feature_pad_mask: パディングマスク
        input_lengths: 入力長
        target_lengths: ターゲット長
        blank_id: ブランクトークンID

    Returns:
        dict: 各層の特徴量辞書
        {
            'cnn_spatial_patterns': CNN出力の空間的パターン特徴量,
            'bilstm_temporal_dynamics': BiLSTM隠れ状態の時系列特徴量,
            'attention_importance_map': Attention重みの重要度マップ,
            'final_integrated_features': 最終層直前の統合特徴量,
            'layer_metadata': 各層のメタデータ
        }
    """
    try:
        model.eval()
        extracted_features = {}
        activations = {}
        layer_metadata = {}

        # フック関数で中間層の出力を取得
        def get_activation(name, description=""):
            def hook(model, input, output):
                # outputがタプルの場合は最初の要素を使用
                if isinstance(output, tuple):
                    actual_output = output[0]
                else:
                    actual_output = output
                
                activations[name] = {
                    'tensor': actual_output.detach() if actual_output is not None else None,
                    'description': description,
                    'shape': actual_output.shape if actual_output is not None else None
                }
                logging.info(f"フック取得: {name} - {description} - 形状: {actual_output.shape if actual_output is not None else None}")
            return hook

        hooks = []

        with torch.no_grad():
            # Attentionの可視化を有効化
            if hasattr(model, 'enable_attention_visualization'):
                model.enable_attention_visualization()

            # 1. CNN出力フック（空間的パターン特徴量）
            if hasattr(model, 'cnn_model'):
                # CNN最終層の出力を取得
                if hasattr(model.cnn_model, 'model') and hasattr(model.cnn_model.model, 'conv1d'):
                    hook = model.cnn_model.model.conv1d.register_forward_hook(
                        get_activation('cnn_spatial_output', 'CNN空間的パターン特徴量（手の形、位置関係、顔の表情、体の姿勢）')
                    )
                    hooks.append(hook)

            # 2. BiLSTMフック（時系列ダイナミクス）
            if hasattr(model, 'rnn') and hasattr(model.rnn, 'bilstm'):
                hook = model.rnn.bilstm.register_forward_hook(
                    get_activation('bilstm_temporal_output', 'BiLSTM時系列ダイナミクス（動きの軌跡、速度、文脈情報）')
                )
                hooks.append(hook)
            elif hasattr(model, 'bilstm'):
                hook = model.bilstm.register_forward_hook(
                    get_activation('bilstm_temporal_output', 'BiLSTM時系列ダイナミクス（動きの軌跡、speed、文脈情報）')
                )
                hooks.append(hook)

            # 3. 空間相関モジュールフック（空間的相関関係）
            if hasattr(model, 'spatial_correlation'):
                hook = model.spatial_correlation.register_forward_hook(
                    get_activation('spatial_correlation_output', '空間相関関係学習（部位間の相互作用）')
                )
                hooks.append(hook)

            # 4. 階層的時間モジュールフック（階層的時間モデリング）
            if hasattr(model, 'hierarchical_temporal'):
                hook = model.hierarchical_temporal.register_forward_hook(
                    get_activation('hierarchical_temporal_output', '階層的時間モデリング（多時間スケール特徴）')
                )
                hooks.append(hook)

            # 5. 分類層直前フック（統合的判断特徴量）
            if hasattr(model, 'classifier'):
                hook = model.classifier.register_forward_hook(
                    get_activation('final_integrated_features', '最終統合特徴量（分類に直結する表現）')
                )
                hooks.append(hook)

            # モデル実行
            logging.info("多層特徴量抽出のためモデルを実行中...")
            forward_result = model.forward(
                src_feature=feature,
                spatial_feature=spatial_feature,
                tgt_feature=tokens,
                src_causal_mask=None,
                src_padding_mask=feature_pad_mask,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                mode="test",
                blank_id=blank_id,
            )

            # 結果を解析
            if len(forward_result) >= 2:
                pred, conv_pred = forward_result[0], forward_result[1]
                if len(forward_result) >= 3:
                    sequence_logits = forward_result[2]
                    layer_metadata['sequence_logits_shape'] = sequence_logits.shape if sequence_logits is not None else None

            # フックを削除
            for hook in hooks:
                hook.remove()

            # 特徴量を整理・変換
            logging.info("特徴量の整理を開始...")
            
            for name, activation_info in activations.items():
                if activation_info['tensor'] is not None:
                    activation = activation_info['tensor']
                    description = activation_info['description']
                    
                    try:
                        # テンソル形状に応じて適切に処理
                        if len(activation.shape) == 4:  # (B, C, T, J) - CNN出力
                            B, C, T, J = activation.shape
                            # 空間次元をフラット化し、時系列で平均化
                            feat = activation.permute(0, 2, 1, 3).contiguous().view(B, T, C * J)
                            # 時系列で平均化して (B, C*J) の特徴量を取得
                            processed_feat = feat.mean(dim=1).cpu().numpy()
                            layer_metadata[f"{name}_original_shape"] = "(B, C, T, J)"
                            layer_metadata[f"{name}_processed_info"] = f"空間特徴量を時系列平均化: {activation.shape} → {processed_feat.shape}"
                            
                        elif len(activation.shape) == 3:  # (B, T, D) または (T, B, D) - BiLSTM出力
                            if activation.shape[1] == feature.shape[0]:  # (T, B, D)
                                feat = activation.permute(1, 0, 2)  # (B, T, D)
                            else:  # (B, T, D)
                                feat = activation
                            # 時系列で平均化
                            processed_feat = feat.mean(dim=1).cpu().numpy()
                            layer_metadata[f"{name}_original_shape"] = str(activation.shape)
                            layer_metadata[f"{name}_processed_info"] = f"時系列特徴量を平均化: {activation.shape} → {processed_feat.shape}"
                            
                        elif len(activation.shape) == 2:  # (B, D) - 最終層など
                            processed_feat = activation.cpu().numpy()
                            layer_metadata[f"{name}_original_shape"] = str(activation.shape)
                            layer_metadata[f"{name}_processed_info"] = f"2次元特徴量をそのまま使用: {activation.shape}"
                            
                        else:
                            # その他の形状は平坦化
                            processed_feat = activation.view(activation.shape[0], -1).cpu().numpy()
                            layer_metadata[f"{name}_original_shape"] = str(activation.shape)
                            layer_metadata[f"{name}_processed_info"] = f"特徴量を平坦化: {activation.shape} → {processed_feat.shape}"

                        # 特徴量を格納
                        layer_key = f"{name}_features"
                        extracted_features[layer_key] = {
                            'features': processed_feat,
                            'description': description,
                            'original_shape': activation.shape,
                            'processed_shape': processed_feat.shape,
                            'layer_type': name.split('_')[0]  # 'cnn', 'bilstm', etc.
                        }
                        
                        logging.info(f"特徴量抽出完了: {layer_key} - {description}")
                        logging.info(f"  形状変換: {activation.shape} → {processed_feat.shape}")
                        
                    except Exception as e:
                        logging.error(f"特徴量処理エラー ({name}): {e}")
                        continue

            # Attention重みの取得（特別処理）
            if hasattr(model, 'get_attention_weights'):
                attention_weights = model.get_attention_weights()
                if attention_weights is not None:
                    try:
                        # Attention重みを重要度マップとして処理
                        if len(attention_weights.shape) == 3:  # (B, T, T)
                            B, T1, T2 = attention_weights.shape
                            # 時系列Attention重みを特徴量として使用
                            # 各時刻での注目度の統計量を特徴量とする
                            attention_stats = np.concatenate([
                                attention_weights.mean(axis=2),  # 平均注目度
                                attention_weights.std(axis=2),   # 注目度の分散
                                attention_weights.max(axis=2)[0], # 最大注目度
                            ], axis=1)
                            
                            extracted_features['attention_importance_map'] = {
                                'features': attention_stats,
                                'description': 'Attention重要度マップ（どの時刻・部位に注目しているか、手話認識の判断根拠）',
                                'original_shape': attention_weights.shape,
                                'processed_shape': attention_stats.shape,
                                'layer_type': 'attention'
                            }
                            
                            layer_metadata['attention_processing'] = f"Attention重み統計特徴量: {attention_weights.shape} → {attention_stats.shape}"
                            logging.info("Attention重要度マップを抽出しました")
                            
                    except Exception as e:
                        logging.error(f"Attention重み処理エラー: {e}")

            # メタデータを格納
            extracted_features['layer_metadata'] = layer_metadata

            logging.info(f"多層特徴量抽出完了: {len(extracted_features)-1}層から特徴量を抽出")
            logging.info(f"抽出した層: {[k for k in extracted_features.keys() if k != 'layer_metadata']}")
            
            return extracted_features

    except Exception as e:
        logging.error(f"多層特徴量抽出でエラー: {e}")
        import traceback
        logging.error(f"詳細なエラー情報: {traceback.format_exc()}")
        return {}


def plot_multilayer_feature_visualization(
    features_dict, labels, vocab_dict=None, save_dir=None, sample_idx=0, method="both"
):
    """
    手話認識の多層特徴量を包括的に可視化
    
    各層の解釈:
    1. CNN出力 → 空間的パターン（手の形、位置関係、顔の表情、体の姿勢）
    2. BiLSTM隠れ状態 → 時系列ダイナミクス（動きの軌跡、速度、手話の文脈情報）
    3. Attention重み → 重要度マップ（どの時刻・部位に注目しているか、判断根拠）
    4. 最終層直前 → 統合的判断（全情報を統合した最終特徴量、分類直結表現）

    Args:
        features_dict: 各層の特徴量辞書
        labels: ラベル配列
        vocab_dict: 語彙辞書
        save_dir: 保存ディレクトリ
        sample_idx: サンプルインデックス
        method: 'tsne', 'umap', 'both'
    """
    try:
        # 必要なライブラリをインポート
        try:
            from sklearn.manifold import TSNE
            import umap
            import seaborn as sns
            from matplotlib.colors import ListedColormap
        except ImportError as e:
            logging.error(f"Feature visualization用ライブラリが不足: {e}")
            logging.info("以下のコマンドでインストールしてください:")
            logging.info("pip install scikit-learn umap-learn seaborn")
            return False

        if not features_dict:
            logging.warning("特徴量データが空です")
            return False

        # メタデータを除外した特徴量層のみを取得
        feature_layers = {k: v for k, v in features_dict.items() if k != 'layer_metadata'}
        
        if not feature_layers:
            logging.warning("有効な特徴量層がありません")
            return False

        # 保存ディレクトリの設定
        if save_dir is None:
            save_dir = config.plot_save_dir
        os.makedirs(save_dir, exist_ok=True)

        # ラベルの準備
        if labels is not None:
            unique_labels = np.unique(labels)
            n_classes = len(unique_labels)
            # カラーマップを設定
            colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
            color_map = dict(zip(unique_labels, colors))
        else:
            unique_labels = [0]
            n_classes = 1
            color_map = {0: 'blue'}

        # 各層の特徴量を順番に可視化
        layer_info_mapping = {
            'cnn': ('CNN空間パターン', '手の形・位置関係・顔の表情・体の姿勢'),
            'bilstm': ('BiLSTM時系列動態', '動きの軌跡・速度・手話の文脈情報'),
            'spatial': ('空間相関関係', '部位間の相互作用・空間的関係性'),
            'hierarchical': ('階層的時間モデル', '多時間スケールでの動き特徴'),
            'attention': ('Attention重要度', 'どの時刻・部位に注目・判断根拠'),
            'final': ('統合的判断特徴', '全情報統合・分類直結表現')
        }

        layer_names = list(feature_layers.keys())
        n_layers = len(layer_names)
        
        logging.info(f"多層特徴量可視化開始: {n_layers}層を処理")

        if method in ["tsne", "both"]:
            # t-SNE可視化
            plt.figure(figsize=(20, 6 * ((n_layers + 2) // 3)))
            
            for i, (layer_name, layer_data) in enumerate(feature_layers.items()):
                features = layer_data['features']
                description = layer_data['description']
                layer_type = layer_data.get('layer_type', 'unknown')
                
                # 特徴量の次元数チェック
                if features.shape[1] < 2:
                    logging.warning(f"{layer_name}: 特徴量次元数が不足 ({features.shape[1]})")
                    continue
                
                # レイヤータイプから説明を取得
                layer_title, layer_detail = layer_info_mapping.get(layer_type, ('不明な層', '詳細不明'))
                
                plt.subplot((n_layers + 2) // 3, 3, i + 1)
                
                try:
                    # t-SNE実行
                    if features.shape[1] > 50:
                        # 高次元の場合はPCAで前処理
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=50)
                        features_reduced = pca.fit_transform(features)
                        explained_var = sum(pca.explained_variance_ratio_)
                        logging.info(f"{layer_name}: PCA前処理 {features.shape[1]}→50次元 (累積寄与率: {explained_var:.3f})")
                    else:
                        features_reduced = features
                    
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]-1))
                    features_2d = tsne.fit_transform(features_reduced)
                    
                    # 散布図プロット
                    if labels is not None:
                        for label in unique_labels:
                            mask = labels == label
                            if np.any(mask):
                                label_name = vocab_dict.get(label, f'Class_{label}') if vocab_dict else f'Class_{label}'
                                plt.scatter(
                                    features_2d[mask, 0], features_2d[mask, 1],
                                    c=[color_map[label]], label=label_name, alpha=0.7, s=50
                                )
                    else:
                        plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.7, s=50)
                    
                    plt.title(f'{layer_title}\n{layer_detail}', fontsize=12, fontweight='bold')
                    plt.xlabel('t-SNE Component 1')
                    plt.ylabel('t-SNE Component 2')
                    
                    if labels is not None and len(unique_labels) <= 10:
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    
                    plt.grid(True, alpha=0.3)
                    
                    # 特徴量の統計情報を追加
                    mean_val = np.mean(features)
                    std_val = np.std(features)
                    plt.text(0.02, 0.98, f'μ={mean_val:.3f}, σ={std_val:.3f}', 
                            transform=plt.gca().transAxes, verticalalignment='top', 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                            fontsize=8)
                    
                    logging.info(f"t-SNE可視化完了: {layer_name} - {features.shape} → 2D")
                    
                except Exception as e:
                    logging.error(f"t-SNE可視化エラー ({layer_name}): {e}")
                    plt.text(0.5, 0.5, f'可視化エラー:\n{str(e)}', ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title(f'{layer_title} (エラー)', fontsize=12)
            
            plt.tight_layout()
            tsne_path = os.path.join(save_dir, f'multilayer_features_tsne_sample_{sample_idx}.png')
            plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"t-SNE多層特徴量可視化を保存: {tsne_path}")

        if method in ["umap", "both"]:
            # UMAP可視化
            plt.figure(figsize=(20, 6 * ((n_layers + 2) // 3)))
            
            for i, (layer_name, layer_data) in enumerate(feature_layers.items()):
                features = layer_data['features']
                description = layer_data['description']
                layer_type = layer_data.get('layer_type', 'unknown')
                
                if features.shape[1] < 2:
                    continue
                
                layer_title, layer_detail = layer_info_mapping.get(layer_type, ('不明な層', '詳細不明'))
                
                plt.subplot((n_layers + 2) // 3, 3, i + 1)
                
                try:
                    # UMAP実行
                    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, features.shape[0]-1))
                    features_2d = reducer.fit_transform(features)
                    
                    # 散布図プロット
                    if labels is not None:
                        for label in unique_labels:
                            mask = labels == label
                            if np.any(mask):
                                label_name = vocab_dict.get(label, f'Class_{label}') if vocab_dict else f'Class_{label}'
                                plt.scatter(
                                    features_2d[mask, 0], features_2d[mask, 1],
                                    c=[color_map[label]], label=label_name, alpha=0.7, s=50
                                )
                    else:
                        plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.7, s=50)
                    
                    plt.title(f'{layer_title}\n{layer_detail}', fontsize=12, fontweight='bold')
                    plt.xlabel('UMAP Component 1')
                    plt.ylabel('UMAP Component 2')
                    
                    if labels is not None and len(unique_labels) <= 10:
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    
                    plt.grid(True, alpha=0.3)
                    
                    # 特徴量の統計情報を追加
                    mean_val = np.mean(features)
                    std_val = np.std(features)
                    plt.text(0.02, 0.98, f'μ={mean_val:.3f}, σ={std_val:.3f}', 
                            transform=plt.gca().transAxes, verticalalignment='top', 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                            fontsize=8)
                    
                    logging.info(f"UMAP可視化完了: {layer_name} - {features.shape} → 2D")
                    
                except Exception as e:
                    logging.error(f"UMAP可視化エラー ({layer_name}): {e}")
                    plt.text(0.5, 0.5, f'可視化エラー:\n{str(e)}', ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title(f'{layer_title} (エラー)', fontsize=12)
            
            plt.tight_layout()
            umap_path = os.path.join(save_dir, f'multilayer_features_umap_sample_{sample_idx}.png')
            plt.savefig(umap_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"UMAP多層特徴量可視化を保存: {umap_path}")

        # 特徴量間相関分析
        plot_feature_correlation_analysis(feature_layers, save_dir, sample_idx)
        
        # 各層の特徴量分布分析
        plot_feature_distribution_analysis(feature_layers, save_dir, sample_idx)
        
        return True

    except Exception as e:
        logging.error(f"多層特徴量可視化でエラー: {e}")
        import traceback
        logging.error(f"詳細なエラー情報: {traceback.format_exc()}")
        return False


def plot_feature_correlation_analysis(feature_layers, save_dir, sample_idx):
    """
    各層間の特徴量相関分析を可視化
    
    Args:
        feature_layers: 特徴量辞書
        save_dir: 保存ディレクトリ
        sample_idx: サンプルインデックス
    """
    try:
        import seaborn as sns
        from scipy.stats import pearsonr
        
        # 各層の特徴量を結合して相関分析
        layer_names = []
        all_features = []
        
        for layer_name, layer_data in feature_layers.items():
            features = layer_data['features']
            # 各層から代表的な特徴量（平均、分散、最大値）を抽出
            layer_summary = np.array([
                np.mean(features, axis=1),
                np.std(features, axis=1),
                np.max(features, axis=1)
            ]).T  # (batch_size, 3)
            
            all_features.append(layer_summary)
            layer_names.extend([f"{layer_name}_mean", f"{layer_name}_std", f"{layer_name}_max"])
        
        if len(all_features) < 2:
            logging.warning("相関分析には最低2層が必要です")
            return
        
        # 特徴量を結合
        combined_features = np.concatenate(all_features, axis=1)  # (batch_size, n_features)
        
        # 相関行列を計算
        correlation_matrix = np.corrcoef(combined_features.T)
        
        # ヒートマップで可視化
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                   xticklabels=layer_names,
                   yticklabels=layer_names,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   linewidths=0.5)
        plt.title('Multi-Layer Feature Correlation Analysis\n各層間の特徴量相関', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        correlation_path = os.path.join(save_dir, f'feature_correlation_sample_{sample_idx}.png')
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"特徴量相関分析を保存: {correlation_path}")
        
    except ImportError:
        logging.warning("seabornが利用できないため、相関分析をスキップします")
    except Exception as e:
        logging.error(f"特徴量相関分析でエラー: {e}")


def plot_feature_distribution_analysis(feature_layers, save_dir, sample_idx):
    """
    各層の特徴量分布を分析・可視化
    
    Args:
        feature_layers: 特徴量辞書
        save_dir: 保存ディレクトリ  
        sample_idx: サンプルインデックス
    """
    try:
        n_layers = len(feature_layers)
        fig, axes = plt.subplots(n_layers, 3, figsize=(15, 5 * n_layers))
        
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        
        for i, (layer_name, layer_data) in enumerate(feature_layers.items()):
            features = layer_data['features']
            description = layer_data['description']
            
            # 1. 特徴量の分布ヒストグラム
            axes[i, 0].hist(features.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[i, 0].set_title(f'{layer_name}\n特徴量分布', fontsize=10, fontweight='bold')
            axes[i, 0].set_xlabel('Feature Value')
            axes[i, 0].set_ylabel('Frequency')
            axes[i, 0].grid(True, alpha=0.3)
            
            # 統計情報を追加
            mean_val = np.mean(features)
            std_val = np.std(features)
            median_val = np.median(features)
            axes[i, 0].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            axes[i, 0].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
            axes[i, 0].legend()
            
            # 2. 次元ごとの特徴量の分散
            feature_vars = np.var(features, axis=0)
            axes[i, 1].plot(range(len(feature_vars)), feature_vars, marker='o', markersize=3)
            axes[i, 1].set_title(f'{layer_name}\n次元別分散', fontsize=10, fontweight='bold')
            axes[i, 1].set_xlabel('Feature Dimension')
            axes[i, 1].set_ylabel('Variance')
            axes[i, 1].grid(True, alpha=0.3)
            
            # 3. サンプル間の特徴量類似度
            if features.shape[0] > 1:
                # コサイン類似度を計算
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(features)
                
                im = axes[i, 2].imshow(similarity_matrix, cmap='viridis', aspect='auto')
                axes[i, 2].set_title(f'{layer_name}\nサンプル間類似度', fontsize=10, fontweight='bold')
                axes[i, 2].set_xlabel('Sample Index')
                axes[i, 2].set_ylabel('Sample Index')
                plt.colorbar(im, ax=axes[i, 2])
            else:
                axes[i, 2].text(0.5, 0.5, 'サンプル数不足', ha='center', va='center', transform=axes[i, 2].transAxes)
                axes[i, 2].set_title(f'{layer_name}\nサンプル間類似度 (N/A)', fontsize=10)
        
        plt.tight_layout()
        distribution_path = os.path.join(save_dir, f'feature_distribution_sample_{sample_idx}.png')
        plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"特徴量分布分析を保存: {distribution_path}")
        
    except ImportError as e:
        logging.warning(f"分布分析に必要なライブラリが不足: {e}")
    except Exception as e:
        logging.error(f"特徴量分布分析でエラー: {e}")


def analyze_feature_separation(features_dict, labels, vocab_dict=None, save_dir=None, sample_idx=0):
    """
    手話認識の各層特徴量の分離度を分析
    
    各層でのクラス分離性能を定量評価し、どの層が分類に有効かを分析
    
    Args:
        features_dict: 各層の特徴量辞書
        labels: クラスラベル
        vocab_dict: 語彙辞書
        save_dir: 保存ディレクトリ
        sample_idx: サンプルインデックス
    """
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import NearestNeighbors
        
        if labels is None:
            logging.warning("ラベルが提供されていないため、分離度分析をスキップします")
            return

        # メタデータを除外した特徴量層のみを取得
        feature_layers = {k: v for k, v in features_dict.items() if k != 'layer_metadata'}
        
        separation_results = {}
        
        for layer_name, layer_data in feature_layers.items():
            features = layer_data['features']
            description = layer_data['description']
            
            try:
                # シルエット係数による分離度評価
                if len(np.unique(labels)) > 1 and features.shape[0] > 1:
                    silhouette_avg = silhouette_score(features, labels)
                    separation_results[layer_name] = {
                        'silhouette_score': silhouette_avg,
                        'description': description,
                        'n_samples': features.shape[0],
                        'n_features': features.shape[1]
                    }
                    
                    logging.info(f"{layer_name}: シルエット係数 = {silhouette_avg:.4f}")
                else:
                    logging.warning(f"{layer_name}: 分離度分析に必要な条件が満たされていません")
                    
            except Exception as e:
                logging.error(f"{layer_name}の分離度分析でエラー: {e}")
                continue
        
        if not separation_results:
            logging.warning("分離度分析結果が得られませんでした")
            return
        
        # 結果を可視化
        plt.figure(figsize=(12, 8))
        
        layer_names = list(separation_results.keys())
        silhouette_scores = [separation_results[name]['silhouette_score'] for name in layer_names]
        
        # 棒グラフで分離度を表示
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(layer_names)), silhouette_scores, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(layer_names))))
        plt.title('Layer-wise Feature Separation Analysis\n各層の特徴量分離度評価', fontsize=14, fontweight='bold')
        plt.xlabel('Layer')
        plt.ylabel('Silhouette Score')
        plt.xticks(range(len(layer_names)), [name.replace('_', '\n') for name in layer_names], rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 各バーに数値を表示
        for i, (bar, score) in enumerate(zip(bars, silhouette_scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 最適な層を特定
        best_layer_idx = np.argmax(silhouette_scores)
        best_layer_name = layer_names[best_layer_idx]
        best_score = silhouette_scores[best_layer_idx]
        
        plt.axhline(y=best_score, color='red', linestyle='--', alpha=0.7, 
                   label=f'Best: {best_layer_name} ({best_score:.3f})')
        plt.legend()
        
        # 詳細分析結果をテキストで表示
        plt.subplot(2, 1, 2)
        plt.axis('off')
        
        # 分析結果のテキスト
        analysis_text = f"""
特徴量分離度分析結果 (Sample {sample_idx})

最適な層: {best_layer_name}
最高シルエット係数: {best_score:.4f}

各層の特徴:
"""
        
        layer_type_meanings = {
            'cnn': 'CNN → 空間的パターン（手の形・位置・表情・姿勢）',
            'bilstm': 'BiLSTM → 時系列ダイナミクス（軌跡・速度・文脈）',
            'spatial': '空間相関 → 部位間相互作用',
            'hierarchical': '階層時間 → 多時間スケール特徴',
            'attention': 'Attention → 重要度マップ（注目箇所・判断根拠）',
            'final': '最終層 → 統合判断（分類直結表現）'
        }
        
        for layer_name, result in separation_results.items():
            layer_type = layer_name.split('_')[0]
            meaning = layer_type_meanings.get(layer_type, '詳細不明')
            analysis_text += f"\n{layer_name}:\n"
            analysis_text += f"  シルエット係数: {result['silhouette_score']:.4f}\n"
            analysis_text += f"  サンプル数: {result['n_samples']}, 特徴量次元: {result['n_features']}\n"
            analysis_text += f"  解釈: {meaning}\n"
        
        # 推奨事項を追加
        analysis_text += f"\n推奨事項:\n"
        if best_score > 0.5:
            analysis_text += f"• {best_layer_name}層は優秀な分離性能を示しています\n"
            analysis_text += f"• この層の特徴量を重点的に活用することを推奨\n"
        elif best_score > 0.25:
            analysis_text += f"• {best_layer_name}層が最も良い分離性能ですが改善の余地があります\n"
            analysis_text += f"• 特徴量エンジニアリングや前処理の見直しを検討\n"
        else:
            analysis_text += f"• 全体的に分離性能が低いです\n"
            analysis_text += f"• モデル構造やデータ前処理の根本的見直しが必要\n"
        
        plt.text(0.05, 0.95, analysis_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_dir is None:
            save_dir = config.plot_save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        separation_path = os.path.join(save_dir, f'feature_separation_analysis_sample_{sample_idx}.png')
        plt.savefig(separation_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"特徴量分離度分析を保存: {separation_path}")
        logging.info(f"=== 分離度分析結果 ===")
        logging.info(f"最適な層: {best_layer_name} (シルエット係数: {best_score:.4f})")
        
        return separation_results
        
    except ImportError as e:
        logging.error(f"分離度分析に必要なライブラリが不足: {e}")
        logging.info("pip install scikit-learn でインストールしてください")
        return None
    except Exception as e:
        logging.error(f"特徴量分離度分析でエラー: {e}")
        import traceback
        logging.error(f"詳細なエラー情報: {traceback.format_exc()}")
        return None
