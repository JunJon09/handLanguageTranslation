import numpy as np
#dataset
read_dataset_dir = "../hdf5/middle_dataset/"
test_number = "003"
val_number = "002"

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

USE_POSE = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]) + 478

USE_LHAND = np.arange(478+33, 478+33+21)
USE_RHAND = np.arange(478+33+21, 478+33+21+21)

use_features = ["x", "y", "z"]
load_into_ram = True
batch_size = 16
spatial_spatial_feature = 25 * 2

#plots
plot_save_dir = "CNN_BiLSTM/reports/figures"
plot_loss_save_path = "cnn_transformer_loss.png"
plot_loss_train_save_path = "cnn_transformer_train_loss.png"
plot_loss_val_save_path = "cnn_transformer_val_loss.png"
plot_loss_train_val_save_path = "cnn_transformer_train_val_loss.png"
plot_wer_save_path = "cnn_transformer_wer.png"