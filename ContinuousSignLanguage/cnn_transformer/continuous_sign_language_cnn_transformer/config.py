import numpy as np

#dataset
read_dataset_dir = "../hdf5/gafs_dataset_very_small/"
test_number = "1"
val_number = "2"

# features 
USE_LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95] 

USE_NOSE = [1, 2, 98, 327]
USE_REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
            246, 161, 160, 159, 158, 157, 173]
USE_LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362,
            466, 388, 387, 386, 385, 384, 398]

USE_LHAND = np.arange(468, 468+21)
USE_POSE = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]) + 468 + 21
# USE_POSE = np.array([15, 16, 17, 18, 19, 20, 21, 22]) + 468 + 21
USE_RHAND = np.arange(468+21+33, 468+21+33+21)

use_features = ["x", "y"]
load_into_ram = True
batch_size = 32

#plots
plot_save_dir = "cnn_transformer/reports/figures"
plot_loss_save_path = "transformer_loss.png"
plot_accuracy_save_path = "transformer_test_accuracy.png"
