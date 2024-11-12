import matplotlib.pyplot as plt
import transformer.continuous_sign_language_transformer.config as config
import numpy as np
import os


def loss_plot(val_losses_trans):

    plt.grid(axis="y", linestyle="dotted", color="k")

    xs = np.arange(1, len(val_losses_trans)+1)
    plt.plot(xs, val_losses_trans, label="Transformer", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim([0.0, 5.0])
    plt.legend()
    save_path = os.path.join(config.plot_save_dir, config.plot_loss_save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_data_plot(test_wers_trans):
    plt.grid(axis="y", linestyle="dotted", color="k")

    xs = np.arange(1, len(test_wers_trans)+1)
    plt.plot(xs, test_wers_trans, label="Transformer", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("WER")
    plt.ylim([0.0, 100.0])
    plt.legend()
    save_path = os.path.join(config.plot_save_dir, config.plot_accuracy_save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()