import os
import transformer.continuous_sign_language_transformer.modeling.config as model_config
import transformer.continuous_sign_language_transformer.modeling.train_functions as functions
import torch
if __name__ == "__main__":
    save_path = os.path.join(model_config.model_save_dir, model_config.model_save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_model, loaded_optimizer, loaded_epoch, loaded_val_losses, test_dataloader, key2token = functions.load_model(save_path, device)

    max_seqlen = 60

    sos_token = key2token["<sos>"]
    eos_token = key2token["<eos>"]

    test_acc = functions.test_loop_csir_s2s(
                test_dataloader, loaded_model, device,
                sos_token, eos_token,
                max_seqlen=max_seqlen,
                return_pred_times=True,
                verbose_num=0)
    print(f"ロードしたモデルのテスト精度: {test_acc}%")
