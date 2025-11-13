import cnn_transformer.continuous_sign_language_cnn_transformer.dataset as dataset
import cnn_transformer.continuous_sign_language_cnn_transformer.modeling.train_functions as functions
import cnn_transformer.continuous_sign_language_cnn_transformer.modeling.config as model_config
import cnn_transformer.models.cnn_transformer as model
import cnn_transformer.continuous_sign_language_cnn_transformer.plots as plot
import torch
import numpy as np
import os

def main():
    train_hdf5files, val_hdf5files, test_hdf5files, key2token = dataset.read_dataset()
    train_dataloader, val_dataloader, test_dataloader, in_channels = functions.set_dataloader(key2token, train_hdf5files, val_hdf5files, test_hdf5files)

    VOCAB = len(key2token)
    out_channels = VOCAB
    pad_token = key2token["<pad>"]

    cnn_transformer = model.CNNTransformerModel(
            in_channels=in_channels,
            inter_channels=model_config.inter_channels,
            kernel_size=model_config.kernel_size,
            stride=model_config.stride,
            out_channels=out_channels,
            padding_val=pad_token,
            activation=model_config.activation,
            tren_num_layers=model_config.tren_num_layers,
            tren_num_heads=model_config.tren_num_heads,
            tren_dim_ffw=model_config.tren_dim_ffw,
            tren_dropout_pe=model_config.tren_dropout_pe,
            tren_dropout=model_config.tren_dropout,
            tren_norm_type_sattn=model_config.norm_type,
            tren_norm_type_ffw=model_config.norm_type,
            tren_norm_type_tail=model_config.norm_type,
            tren_norm_eps=model_config.tren_norm_eps,
            tren_norm_first=model_config.tren_norm_first,
            tren_add_bias=model_config.tren_add_bias,
            tren_add_tailnorm=model_config.tren_add_tailnorm,
            trde_num_layers=model_config.trde_num_layers,
            trde_num_heads=model_config.trde_num_heads,
            trde_dim_ffw=model_config.trde_dim_ffw,
            trde_dropout_pe=model_config.trde_dropout_pe,
            trde_dropout=model_config.trde_dropout,
            trde_norm_type_sattn=model_config.norm_type,
            trde_norm_type_cattn=model_config.norm_type,
            trde_norm_type_ffw=model_config.norm_type,
            trde_norm_type_tail=model_config.norm_type,
            trde_norm_eps=model_config.trde_norm_eps,
            trde_norm_first=model_config.trde_norm_first,
            trde_add_bias=model_config.trde_add_bias,
            trde_add_tailnorm=model_config.trde_add_tailnorm)

    label_smoothing = model_config.label_smoothing
    lr = model_config.lr
    epochs = model_config.epochs
    eval_every_n_epochs = model_config.eval_every_n_epochs
    max_seqlen = model_config.max_seqlen
    sos_token = key2token["<sos>"]
    eos_token = key2token["<eos>"]
    pad_token = key2token["<pad>"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = functions.LabelSmoothingCrossEntropyLoss(
    ignore_indices=pad_token, reduction="mean_temporal_prior",
    label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(cnn_transformer.parameters(), lr=lr)

    cnn_transformer.to(device)

    train_losses = []
    val_losses = []
    test_wers = []
    MIN_WER = float('inf')
    save_path = os.path.join(model_config.model_save_dir, model_config.model_save_path)
    print("Start training.")
    for epoch in range(epochs):
        print("-" * 80)
        print(f"Epoch {epoch+1}")

        train_loss, train_times = functions.train_loop_csir_s2s(
            train_dataloader, cnn_transformer, loss_fn, optimizer, device,
            sos_token, eos_token,
            return_pred_times=True)
        val_loss, val_times = functions.val_loop_csir_s2s(
            val_dataloader, cnn_transformer, loss_fn, device,
            sos_token, eos_token,
            return_pred_times=True)
        val_losses.append(val_loss)

        if (epoch+1) % eval_every_n_epochs == 0:
            wer, test_times = functions.test_loop_csir_s2s(
                test_dataloader, cnn_transformer, device,
                sos_token, eos_token,
                max_seqlen=max_seqlen,
                return_pred_times=True,
                verbose_num=0)
            test_wers.append(wer)
            if MIN_WER > wer:
                print("最高値更新", wer)
                functions.save_model(save_path, model_default_dict=cnn_transformer.state_dict(), optimizer_dict=optimizer.state_dict(), epoch=model_config.epochs)
                MIN_WER = wer
    train_losses_trans = np.array(train_losses)
    val_losses_trans = np.array(val_losses)
    test_wers_trans = np.array(test_wers)

    val_losses_trans = np.array(val_losses_trans)
    test_wers_trans = np.array(test_wers_trans)
    print(f"Minimum validation loss:{val_losses_trans.min()} at {np.argmin(val_losses_trans)+1} epoch.")
    print(f"Minimum WER:{test_wers_trans.min()} at {np.argmin(test_wers_trans)*eval_every_n_epochs+1} epoch.")

    print(val_losses_trans)
    plot.loss_plot(val_losses_trans)
    plot.test_data_plot(test_wers_trans)

if __name__ == "__main__":
    main()