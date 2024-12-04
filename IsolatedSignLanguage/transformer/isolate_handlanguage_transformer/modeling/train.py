import transformer.isolate_handlanguage_transformer.dataset as dataset
import transformer.isolate_handlanguage_transformer.plots as plot
import transformer.models.model as model
import transformer.isolate_handlanguage_transformer.modeling.config as model_config
import transformer.isolate_handlanguage_transformer.modeling.train_functions as functions
import os
from torch import nn
import torch
import numpy as np

def main(test_number, val_number, model_save_path):
    train_hdf5files, val_hdf5files, test_hdf5files, VOCAB = dataset.read_dataset(test_number, val_number)
    train_dataloader, val_dataloader, test_dataloader, in_channels = functions.set_dataloader(train_hdf5files, val_hdf5files, test_hdf5files)

    out_channels = VOCAB

    model_default = model.TransformerModel(
        in_channels=in_channels,
        inter_channels=model_config.inter_channels,
        out_channels=out_channels,
        activation=model_config.activation,
        tren_num_layers=model_config.tren_num_layers,
        tren_num_heads=model_config.tren_num_heads,
        tren_dim_ffw=model_config.tren_dim_ffw,
        tren_dropout_pe=model_config.tren_dropout_pe,
        tren_dropout=model_config.tren_dropout,
        tren_layer_norm_eps=model_config.tren_layer_norm_eps,
        tren_norm_first=model_config.tren_norm_first,
        tren_add_bias=model_config.tren_add_bias,
        tren_add_tailnorm=model_config.tren_add_tailnorm)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(model_default.parameters(), lr=model_config.lr)

    model_default.to(device)
    train_losses = []
    val_losses = []
    test_accs = []
    print("Start training.")
    for epoch in range(model_config.epochs):
        print("-" * 80)
        print(f"Epoch {epoch+1}")

        train_losses = functions.train_loop(train_dataloader, model_default, loss_fn, optimizer, device)
        val_loss = functions.val_loop(val_dataloader, model_default, loss_fn, device)
        val_losses.append(val_loss)

        if (epoch+1) % model_config.eval_every_n_epochs == 0:
            acc = functions.test_loop(test_dataloader, model_default, device)
            test_accs.append(acc)
    train_losses_default = np.array(train_losses)
    val_losses_default = np.array(val_losses)
    test_accs_default = np.array(test_accs)

    print(f"Minimum validation loss:{val_losses_default.min()} at {np.argmin(val_losses_default)+1} epoch.")
    print(f"Maximum accuracy:{test_accs_default.max()} at {np.argmax(test_accs_default)*model_config.eval_every_n_epochs+1} epoch.")


    save_path = os.path.join(model_config.model_save_dir, model_save_path)
    functions.save_model(save_path, model_default_dict=model_default.state_dict(), optimizer_dict=optimizer.state_dict(), epoch=model_config.epochs, val_loss=val_losses_default)

    plot.loss_plot(val_losses_default=val_losses_default)
    plot.test_data_plot(test_accs_default=test_accs_default)



if __name__ == "__main__":
    for test in range(1, 11):
        for val in range(1, 11):
            if val == test:
                continue
            val_number = str(val).zfill(3)
            test_number = str(test).zfill(3)
            model_path = "transformer_model_val_" + val_number + "_test_" + test_number + ".pth"
            print(model_path)
            main(test_number, val_number, model_path)