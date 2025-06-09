import matplotlib.pyplot as plt
import numpy as np
import os
import CNN_BiLSTM.continuous_sign_language.config as config
from collections import Counter

def train_loss_plot(losses_default):

    plt.grid(axis="y", linestyle="dotted", color="k")

    xs = np.arange(1, len(losses_default)+1)
    plt.plot(xs, losses_default, label="Default", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim([0.0, 5.0])
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(config.plot_save_dir, config.plot_loss_train_save_path)
    print(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def val_loss_plot(val_losses_default, eval_every_n_epochs):
    plt.grid(axis="y", linestyle="dotted", color="red")

    xs = np.arange(1, len(val_losses_default)*eval_every_n_epochs+1)
    xs_val = np.arange(eval_every_n_epochs, len(val_losses_default)*eval_every_n_epochs+1, eval_every_n_epochs)
    plt.plot(xs_val, val_losses_default, label="Default", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim([0.0, 50.0])
    plt.xticks(np.arange(1, len(val_losses_default)+1, eval_every_n_epochs))
    plt.legend()
    save_path = os.path.join(config.plot_save_dir, config.plot_loss_val_save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
def test_data_plot(test_accs_default):
    plt.grid(axis="y", linestyle="dotted", color="k")

    xs = np.arange(1, len(test_accs_default)+1)
    plt.plot(xs, test_accs_default, label="Default", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim([0.0, 100.0])
    plt.legend()
    save_path = os.path.join(config.plot_save_dir, config.plot_accuracy_save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def wer_plot(wer_scores_default, eval_every_n_epochs):
    plt.grid(axis="y", linestyle="dotted", color="red")

    xs = np.arange(1, len(wer_scores_default)*eval_every_n_epochs+1)
    xs_val = np.arange(eval_every_n_epochs, len(wer_scores_default)*eval_every_n_epochs+1, eval_every_n_epochs)
    plt.plot(xs_val, wer_scores_default, label="Default", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("WER")
    plt.ylim([0.0, 1.5])
    plt.xticks(np.arange(1, len(wer_scores_default)+1, eval_every_n_epochs))
    plt.legend()
    save_path = os.path.join(config.plot_save_dir, config.plot_wer_save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, token in enumerate(top5_tokens):
        plt.plot(batch_numbers, token_trends[token], 
                label=f'{token}', 
                color=colors[i], 
                marker=markers[i], 
                markersize=6,
                linewidth=2,
                alpha=0.8)
    
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Average Probability (Non-blank)', fontsize=12)
    plt.title('Top 5 Non-blank Token Trends', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 統計情報を表示
    print(f"\n=== Top 5 非ブランクトークン分析 (バッチ数: {len(self.token_frequencies)}) ===")
    for i, token in enumerate(top5_tokens):
        final_prob = token_trends[token][-1] if token_trends[token] else 0
        max_prob = max(token_trends[token]) if token_trends[token] else 0
        avg_prob = np.mean(token_trends[token]) if token_trends[token] else 0
        print(f"{i+1}. {token}:")
        print(f"   最新: {final_prob:.4f}, 最大: {max_prob:.4f}, 平均: {avg_prob:.4f}")
    
    if save_plot:
        # 保存先ディレクトリの設定
        if plot_dir is None:
            plot_dir = config.plot_save_dir if hasattr(config, 'plot_save_dir') else '.'
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(plot_dir, exist_ok=True)
        
        # ファイル名に日時を追加してユニークにする
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(plot_dir, f'non_blank_token_trends_{timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nグラフを保存しました: {save_path}")
    
    # plt.show() は削除
    plt.close()  # メモリリークを防ぐために明示的に閉じる