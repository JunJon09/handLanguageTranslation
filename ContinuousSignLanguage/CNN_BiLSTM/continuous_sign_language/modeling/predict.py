import CNN_BiLSTM.continuous_sign_language.modeling.functions as functions
import CNN_BiLSTM.models.cnn_bilstm_model as model
import CNN_BiLSTM.continuous_sign_language.modeling.config as model_config
import CNN_BiLSTM.continuous_sign_language.dataset as dataset
import torch

if __name__ == "__main__":
    train_hdf5files, val_hdf5files, test_hdf5files, key2token = dataset.read_dataset()
    train_dataloader, val_dataloader, test_dataloader, in_channels = functions.set_dataloader(key2token, train_hdf5files, val_hdf5files, test_hdf5files)
    VOCAB = len(key2token)
    out_channels = VOCAB
    save_path = model_config.model_save_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cnn_transformer = model.CNNBiLSTMModel(
        in_channels=in_channels,
        kernel_size=model_config.kernel_size,
        cnn_out_channels=model_config.cnn_out_channels,
        stride=model_config.stride,
        padding=model_config.padding,
        dropout_rate=model_config.dropout_rate,
        bias=model_config.bias,
        resNet=model_config.resNet,
        activation=model_config.activation,
        tren_num_layers=model_config.tren_num_layers,
        tren_num_heads=model_config.tren_num_heads,       
        tren_dim_ffw=model_config.tren_dim_ffw,    
        tren_dropout=model_config.tren_dropout,
        tren_norm_eps=model_config.tren_norm_eps,
        batch_first=model_config.batch_first,
        tren_norm_first=model_config.tren_norm_first,
        tren_add_bias=model_config.tren_add_bias,
        num_classes=out_channels,
        blank_idx=VOCAB-1,
    )

    load_model, optimizer_loaded, epoch_loaded = functions.load_model(cnn_transformer, save_path, device)


    wer, test_times = functions.test_loop(dataloader=test_dataloader, model=load_model, device=device, return_pred_times=True, blank_id=VOCAB-1)
    print(f"ロードしたモデルのテスト精度: {wer}")
