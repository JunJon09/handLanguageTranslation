import one_dcnn_transformer_encoder.continuous_sign_language.dataset as dataset
import one_dcnn_transformer_encoder.continuous_sign_language.modeling.functions as functions
import one_dcnn_transformer_encoder.models.one_dcnn_transformer_encoder as model
import one_dcnn_transformer_encoder.continuous_sign_language.modeling.config as model_config
import one_dcnn_transformer_encoder.continuous_sign_language.plots as plot
import torch
import os
import numpy as np

def model_train():
    train_hdf5files, val_hdf5files, test_hdf5files, key2token = dataset.read_dataset()
    train_dataloader, val_dataloader, test_dataloader, in_channels = functions.set_dataloader(key2token, train_hdf5files, val_hdf5files, test_hdf5files)

    VOCAB = len(key2token)
    out_channels = VOCAB
    pad_token = key2token["<pad>"]

    cnn_transformer = model.OnedCNNTransformerEncoderModel(
        in_channels=in_channels,
        kernel_size=model_config.kernel_size,
        inter_channels=model_config.inter_channels, 
        stride=model_config.stride,
        padding=model_config.padding,
        activation=model_config.activation,
        tren_num_layers=model_config.tren_num_layers,
        tren_num_heads=model_config.tren_num_heads,       
        tren_dim_ffw=model_config.tren_dim_ffw,    
        tren_dropout=model_config.tren_dropout,
        tren_norm_eps=model_config.tren_norm_eps,
        batch_first=model_config.batch_first,
        tren_norm_first=model_config.tren_norm_first,
        tren_add_bias=model_config.tren_add_bias,
        num_classes=100,
        blank_idx=VOCAB - 1,
    )

    #初期値を確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(cnn_transformer.parameters(), lr=model_config.lr)
    epochs = model_config.epochs
    max_seqlen = model_config.max_seqlen
    train_losses = []
    val_losses = []
    min_loss = float('inf')
    cnn_transformer.to(device)

    print("Start training.")
    for epoch in range(epochs):
        print("-" * 80)
        print(f"Epoch {epoch+1}")
        train_loss, train_times = functions.train_loop(
            dataloader=train_dataloader, model=cnn_transformer, optimizer=optimizer, device=device, return_pred_times=True)
        train_losses.append(train_loss)
        
        if (epoch + 1) % model_config.eval_every_n_epochs == 0:
            val_loss, val_times = functions.val_loop(
                dataloader=val_dataloader, model=cnn_transformer, device=device, return_pred_times=True)
            val_losses.append(val_loss)
            if min_loss > val_loss: #lossが最小なのを保存
                functions.save_model(
                    save_path=model_config.model_save_path,model_default_dict=cnn_transformer.state_dict(), optimizer_dict=optimizer.state_dict(), epoch=model_config.epochs)
                min_loss = val_loss
    
    train_losses_array = np.array(train_losses)
    val_losses_array = np.array(val_losses)
    print(f"Minimum validation loss:{val_losses_array.min()} at {np.argmin(val_losses_array)+1} epoch.")
    
    plot.train_loss_plot(train_losses_array)
    


if __name__ == "__main__":
    model_train()