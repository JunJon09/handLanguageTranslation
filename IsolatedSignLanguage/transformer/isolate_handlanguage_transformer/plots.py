import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = "transformer/reports/figures"

def loss_plot(val_losses_default):

    plt.grid(axis="y", linestyle="dotted", color="k")

    xs = np.arange(1, len(val_losses_default)+1)
    plt.plot(xs, val_losses_default, label="Default", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim([0.0, 2.5])
    plt.legend()
    save_path = os.path.join(save_dir, "transformer_loss.pdf")
    plt.savefig(save_path, format="pdf", facecolor='w')
    plt.close()

def test_data_plot(test_accs_default):
    plt.grid(axis="y", linestyle="dotted", color="k")

    xs = np.arange(1, len(test_accs_default)+1)
    plt.plot(xs, test_accs_default, label="Default", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim([0.0, 100.0])
    plt.legend()
    save_path = os.path.join(save_dir, "transformer_test_accuracy.pdf")
    plt.savefig(save_path, format="pdf", facecolor='w')
    plt.close()