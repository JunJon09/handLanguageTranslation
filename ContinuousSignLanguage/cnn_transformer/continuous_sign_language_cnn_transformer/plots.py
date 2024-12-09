import matplotlib.pyplot as plt
import numpy as np
import os
import cnn_transformer.continuous_sign_language_cnn_transformer.config as config

def loss_plot(val_losses_default):

    plt.grid(axis="y", linestyle="dotted", color="k")

    xs = np.arange(1, len(val_losses_default)+1)
    plt.plot(xs, val_losses_default, label="Default", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim([0.0, 2.5])
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(config.plot_save_dir, config.plot_loss_save_path)
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
