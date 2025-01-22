import one_dcnn_transformer_encoder.continuous_sign_language.dataset as dataset

def model_train():
    train_hdf5files, val_hdf5files, test_hdf5files, key2token = dataset.read_dataset()
    

if __name__ == "__main__":
    model_train()