from pathlib import Path
import numpy as np

# Load environment variables from .env file if it exists


# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

#dataset
read_dataset_dir = "../hdf5/lsa64_liner/"
test_number = "001"
val_number = "002"

#train
use_features  = ["x", "y"]
batch_size = 32
load_into_ram = True

# features
USE_LIP = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80,
            81, 82, 84, 87, 88, 91, 95, 146, 178, 181,
            185, 191, 267, 269, 270, 291, 308, 310, 311, 312,
            314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
USE_NOSE = [1, 2, 98, 327]
USE_REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                246, 161, 160, 159, 158, 157, 173]
USE_LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362,
                466, 388, 387, 386, 385, 384, 398]

USE_LHAND = np.arange(468, 468+21)
USE_POSE = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]) + 468 + 21
# USE_POSE = np.array([15, 16, 17, 18, 19, 20, 21, 22]) + 468 + 21
USE_RHAND = np.arange(468+21+33, 468+21+33+21)

#plots
plot_save_dir = "transformer/reports/figures"
plot_loss_save_path = "transformer_liner_loss.png"
plot_accuracy_save_path = "transformer_test_liner_accuracy.png"

