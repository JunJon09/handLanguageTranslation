import transformer.continuous_sign_language_transformer.dataset as dataset
if __name__ == "__main__":
    train_hdf5files, val_hdf5files, test_hdf5files, VOCAB = dataset.read_dataset()
    