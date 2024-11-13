import transformer.continuous_sign_language_transformer.dataset as dataset
import transformer.continuous_sign_language_transformer.modeling.train_functions as functions
import transformer.continuous_sign_language_transformer.modeling.config as model_config
import transformer.models.cslr_model as models
import transformer.continuous_sign_language_transformer.plots as plot
import os
import torch
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    train_hdf5files, val_hdf5files, test_hdf5files, key2token = dataset.read_dataset()
    train_dataloader, val_dataloader, test_dataloader, in_channels = functions.set_dataloader(key2token, train_hdf5files, val_hdf5files, test_hdf5files)

    VOCAB = len(key2token)
    out_channels = VOCAB
    pad_token = key2token["<pad>"]


    model_transformer = models.TransformerCSLR(
    in_channels=in_channels,
    inter_channels=model_config.inter_channels,
    out_channels=out_channels,
    padding_val=pad_token,
    activation="relu",
    tren_num_layers=2,
    tren_num_heads=2,
    tren_dim_ffw=256,
    tren_dropout_pe=0.1,
    tren_dropout=0.1,
    tren_norm_type_sattn=model_config.norm_type,
    tren_norm_type_ffw=model_config.norm_type,
    tren_norm_type_tail=model_config.norm_type,
    tren_norm_eps=1e-5,
    tren_norm_first=True,
    tren_add_bias=True,
    tren_add_tailnorm=True,
    trde_num_layers=2,
    trde_num_heads=2,
    trde_dim_ffw=256,
    trde_dropout_pe=0.1,
    trde_dropout=0.1,
    trde_norm_type_sattn=model_config.norm_type,
    trde_norm_type_cattn=model_config.norm_type,
    trde_norm_type_ffw=model_config.norm_type,
    trde_norm_type_tail=model_config.norm_type,
    trde_norm_eps=1e-5,
    trde_norm_first=True,
    trde_add_bias=True,
    trde_add_tailnorm=True)

    label_smoothing = 0.1
    lr = 3e-4
    epochs = 50
    eval_every_n_epochs = 1
    max_seqlen = 60
    sos_token = key2token["<sos>"]
    eos_token = key2token["<eos>"]
    pad_token = key2token["<pad>"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = functions.LabelSmoothingCrossEntropyLoss(
    ignore_indices=pad_token, reduction="mean_temporal_prior",
    label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model_transformer.parameters(), lr=lr)

    model_transformer.to(device)

    train_losses = []
    val_losses = []
    test_wers = []
    print("Start training.")
    for epoch in range(epochs):
        print("-" * 80)
        print(f"Epoch {epoch+1}")

        train_loss, train_times = functions.train_loop_csir_s2s(
            train_dataloader, model_transformer, loss_fn, optimizer, device,
            sos_token, eos_token,
            return_pred_times=True)
        val_loss, val_times = functions.val_loop_csir_s2s(
            val_dataloader, model_transformer, loss_fn, device,
            sos_token, eos_token,
            return_pred_times=True)
        val_losses.append(val_loss)

        if (epoch+1) % eval_every_n_epochs == 0:
            wer, test_times = functions.test_loop_csir_s2s(
                test_dataloader, model_transformer, device,
                sos_token, eos_token,
                max_seqlen=max_seqlen,
                return_pred_times=True,
                verbose_num=0)
            test_wers.append(wer)
    train_losses_trans = np.array(train_losses)
    val_losses_trans = np.array(val_losses)
    test_wers_trans = np.array(test_wers)

    val_losses_trans = np.array(val_losses_trans)
    test_wers_trans = np.array(test_wers_trans)
    print(f"Minimum validation loss:{val_losses_trans.min()} at {np.argmin(val_losses_trans)+1} epoch.")
    print(f"Minimum WER:{test_wers_trans.min()} at {np.argmin(test_wers_trans)*eval_every_n_epochs+1} epoch.")

    save_path = os.path.join(model_config.model_save_dir, model_config.model_save_path)
    functions.save_model(save_path, model_default_dict=model_transformer.state_dict(), optimizer_dict=optimizer.state_dict(), epoch=model_config.epochs, val_loss=val_losses_trans)

    plot.loss_plot(val_losses_trans=val_losses_trans)
    plot.test_data_plot(test_wers_trans=test_wers_trans)