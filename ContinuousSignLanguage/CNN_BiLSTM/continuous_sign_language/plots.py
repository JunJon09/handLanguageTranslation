import matplotlib.pyplot as plt
import numpy as np
import os
import CNN_BiLSTM.continuous_sign_language.config as config
from collections import Counter
import logging


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


def plot_word_error_distribution(word_error_counts):
    """
    Visualize word-level error counts with bar charts

    Args:
        word_error_counts: Dictionary of word-level error statistics
            Example: {'word1': {'correct': 10, 'incorrect': 5, 'total': 15}, ...}
        save_path: Save path (uses config.plot_save_dir if None)
        top_n: Number of top words to display (default: 30)

    Returns:
        bool: Whether visualization succeeded
    """
    try:
        if not word_error_counts:
            logging.warning("Word error statistics data is empty")
            return False

        logging.info(
            f"Starting word error distribution visualization ({len(word_error_counts)} words total)"
        )

        # Sort by error count
        sorted_words = sorted(
            word_error_counts.items(), key=lambda x: x[1]["incorrect"], reverse=True
        )

        # Get top N words
        top_words = sorted_words[:config.top_n]

        # Extract data
        words = [word for word, _ in top_words]
        incorrect_counts = [stats["incorrect"] for _, stats in top_words]
        correct_counts = [stats["correct"] for _, stats in top_words]
        error_rates = [
            stats["incorrect"] / stats["total"] if stats["total"] > 0 else 0
            for _, stats in top_words
        ]

        # Set figure size
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # === Top panel: Stacked bar chart (correct/incorrect breakdown) ===
        x_pos = np.arange(len(words))
        width = 0.8

        # Stack correct and incorrect
        bars1 = ax1.bar(
            x_pos,
            incorrect_counts,
            width,
            label="Incorrect",
            color="#e74c3c",
            alpha=0.8,
        )
        bars2 = ax1.bar(
            x_pos,
            correct_counts,
            width,
            bottom=incorrect_counts,
            label="Correct",
            color="#2ecc71",
            alpha=0.8,
        )

      
        ax1.set_ylabel("Occurrence Count", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"Word Recognition Accuracy (Top {config.top_n} Words by Error Count)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(words, rotation=45, ha="right", fontsize=10)
        ax1.legend(loc="upper right", fontsize=11)
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        # Display total count above each bar
        for i, (incorrect, correct) in enumerate(zip(incorrect_counts, correct_counts)):
            total = incorrect + correct
            ax1.text(i, total + 0.5, str(total), ha="center", va="bottom", fontsize=9)

        colors = [
            "#e74c3c" if rate > 0.5 else "#f39c12" if rate > 0.3 else "#3498db"
            for rate in error_rates
        ]
        bars3 = ax2.bar(
            x_pos, error_rates, width, color=colors, alpha=0.8, edgecolor="black"
        )
     
        ax2.set_xlabel("Word", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Error Rate", fontsize=12, fontweight="bold")
        ax2.set_title("Error Rate by Word", fontsize=14, fontweight="bold", pad=20)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(words, rotation=45, ha="right", fontsize=10)
        ax2.set_ylim([0, 1.0])
        ax2.axhline(
            y=0.5,
            color="red",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
            label="50% Line",
        )
        ax2.axhline(
            y=0.3,
            color="orange",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
            label="30% Line",
        )
        ax2.legend(loc="upper right", fontsize=10)
        ax2.grid(axis="y", alpha=0.3, linestyle="--")

        # Display error rate above each bar
        for i, rate in enumerate(error_rates):
            ax2.text(
                i, rate + 0.02, f"{rate:.1%}", ha="center", va="bottom", fontsize=9
            )

        # Add statistics
        total_words = len(word_error_counts)
        total_incorrect = sum(
            stats["incorrect"] for stats in word_error_counts.values()
        )
        total_correct = sum(stats["correct"] for stats in word_error_counts.values())
        total_occurrences = total_incorrect + total_correct
        overall_error_rate = (
            total_incorrect / total_occurrences if total_occurrences > 0 else 0
        )

        stats_text = f"Statistics:\n"
        stats_text += f"• Total words: {total_words}\n"
        stats_text += f"• Total occurrences: {total_occurrences}\n"
        stats_text += f"• Total errors: {total_incorrect}\n"
        stats_text += f"• Total correct: {total_correct}\n"
        stats_text += f"• Overall error rate: {overall_error_rate:.2%}\n"
        stats_text += f"• Most error word: '{words[0]}' ({incorrect_counts[0]} errors)"

        fig.text(
            0.02,
            0.48,
            stats_text,
            fontsize=10,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.8),
            verticalalignment="center",
        )

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = os.path.join(config.plot_save_dir, config.word_error_distribution_save_path)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Saved word error distribution chart: {save_path}")
        logging.info(
            f"Overall error rate: {overall_error_rate:.2%} ({total_incorrect}/{total_occurrences})"
        )

        return True

    except Exception as e:
        logging.error(f"Error in word error distribution visualization: {e}")
        import traceback

        logging.error(f"Detailed error info: {traceback.format_exc()}")
        return False
