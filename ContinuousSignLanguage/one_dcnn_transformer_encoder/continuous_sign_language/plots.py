import matplotlib.pyplot as plt
import numpy as np
import os
import one_dcnn_transformer_encoder.continuous_sign_language.config as config

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
    plt.ylim([0.0, 5.0])
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
    plt.ylim([0.0, 5.0])
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
