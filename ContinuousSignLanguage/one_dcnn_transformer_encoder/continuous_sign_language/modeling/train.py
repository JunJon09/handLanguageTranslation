import one_dcnn_transformer_encoder.continuous_sign_language.dataset as dataset
import one_dcnn_transformer_encoder.continuous_sign_language.modeling.functions as functions
def model_train():
    train_hdf5files, val_hdf5files, test_hdf5files, key2token = dataset.read_dataset()
    train_dataloader, val_dataloader, test_dataloader, in_channels = functions.set_dataloader(key2token, train_hdf5files, val_hdf5files, test_hdf5files)

    VOCAB = len(key2token)
    out_channels = VOCAB
    pad_token = key2token["<pad>"]

    


if __name__ == "__main__":
    model_train()