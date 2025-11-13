from typing import Tuple
import torch
import transformer.isolate_handlanguage_transformer.modeling.train_functions as functions
import transformer.isolate_handlanguage_transformer.modeling.config as model_config
import os


if __name__ == "__main__":
    save_path = os.path.join(model_config.model_save_dir, model_config.model_save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_model, loaded_optimizer, loaded_epoch, loaded_val_losses, test_dataloader = functions.load_model(save_path, device)

    test_acc = functions.test_loop(test_dataloader, loaded_model, device)
    print(f"ロードしたモデルのテスト精度: {test_acc}%")