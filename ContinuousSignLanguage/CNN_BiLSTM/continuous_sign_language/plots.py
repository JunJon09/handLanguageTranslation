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
        logging.debug(f"moving_avg.shape: {moving_avg.shape}, x_smooth.shape: {x_smooth.shape}")
        
        # 長さが一致していることを確認
        if len(x_smooth) != len(moving_avg):
            logging.error(f"x_smooth length ({len(x_smooth)}) != moving_avg length ({len(moving_avg)})")
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
    y_true, y_pred, class_names=None, save_path=None, 
    normalize=True, title="Confusion Matrix", figsize=(12, 10)
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
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # NaNを0に変換
        else:
            cm_normalized = cm
            
        # 図を作成
        plt.figure(figsize=figsize)
        
        # クラス名の設定
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        # ヒートマップを描画
        if normalize:
            sns.heatmap(
                cm_normalized, 
                annot=True, 
                fmt='.3f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'}
            )
            plt.title(f'{title} (Normalized)')
        else:
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'}
            )
            plt.title(f'{title} (Raw Counts)')
            
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 統計情報を追加
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f}', 
                   fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(config.plot_save_dir, "confusion_matrix.png")
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
            logging.info(f"{class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # 最も間違えやすいペアを特定
        logging.info("\n=== 主な誤分類パターン ===")
        confusion_pairs = []
        
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and cm[i, j] > 0:
                    true_class = class_names[i] if i < len(class_names) else f"Class_{i}"
                    pred_class = class_names[j] if j < len(class_names) else f"Class_{j}"
                    confusion_pairs.append((cm[i, j], true_class, pred_class))
        
        # 間違いの多い順にソート
        confusion_pairs.sort(reverse=True)
        
        # 上位5つの誤分類パターンを表示
        for count, true_cls, pred_cls in confusion_pairs[:5]:
            logging.info(f"{true_cls} -> {pred_cls}: {count}回")
            
    except Exception as e:
        logging.error(f"混同行列分析でエラー: {e}")


def analyze_word_level_confusion(predictions, ground_truth, vocab_dict=None, save_path=None):
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
            figsize=(15, 12)
        )
        
        if success:
            logging.info("単語レベル混同行列分析が完了")
        
        return success
        
    except Exception as e:
        logging.error(f"単語レベル混同行列分析でエラー: {e}")
        return False
