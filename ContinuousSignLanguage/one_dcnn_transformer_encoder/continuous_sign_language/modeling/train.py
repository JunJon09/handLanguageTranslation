import one_dcnn_transformer_encoder.continuous_sign_language.dataset as dataset
import one_dcnn_transformer_encoder.continuous_sign_language.modeling.functions as functions
import one_dcnn_transformer_encoder.models.one_dcnn_transformer_encoder as model
import one_dcnn_transformer_encoder.continuous_sign_language.modeling.config as model_config
import torch

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    model_train()