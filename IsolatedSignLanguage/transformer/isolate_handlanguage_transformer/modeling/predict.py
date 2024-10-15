from typing import Tuple
import torch
import transformer.models.model as model
import numpy as np
import transformer.isolate_handlanguage_transformer.modeling.train as train
import transformer.isolate_handlanguage_transformer.dataset as dataset
import os

def load_model(save_path: str, device: str = "cpu") -> Tuple[model.TransformerModel, torch.optim.Adam, int, np.ndarray, list]:
    """
    保存されたモデルとオプティマイザをロードします。

    Args:
        save_path (str): 保存されたファイルのパス。
        device (str, optional): デバイス。デフォルトは "cpu"。

    Returns:
        Tuple[model.TransformerModel, torch.optim.Adam, int, np.ndarray]: ロードされたモデル、オプティマイザ、エポック数、検証損失。
    """
    checkpoint = torch.load(save_path, map_location=device)

    train_hdf5files, val_hdf5files, test_hdf5files, VOCAB = dataset.read_dataset()
    train_dataloader, val_dataloader, test_dataloader, in_channels = train.set_dataloader(train_hdf5files, val_hdf5files, test_hdf5files)
    
    # モデルの再構築
    inter_channels = 64  # 保存時と同じハイパーパラメータを使用
    out_channels = VOCAB
    activation = "relu"
    tren_num_layers = 6
    tren_num_heads = 2
    tren_dim_ffw = 256
    tren_dropout_pe = 0.1
    tren_dropout = 0.1
    tren_layer_norm_eps = 1e-5
    tren_norm_first = False
    tren_add_bias = True
    tren_add_tailnorm = False

    model_loaded = model.TransformerModel(
        in_channels=in_channels,
        inter_channels=inter_channels,
        out_channels=out_channels,
        activation=activation,
        tren_num_layers=tren_num_layers,
        tren_num_heads=tren_num_heads,
        tren_dim_ffw=tren_dim_ffw,
        tren_dropout_pe=tren_dropout_pe,
        tren_dropout=tren_dropout,
        tren_layer_norm_eps=tren_layer_norm_eps,
        tren_norm_first=tren_norm_first,
        tren_add_bias=tren_add_bias,
        tren_add_tailnorm=tren_add_tailnorm)
    
    model_loaded.load_state_dict(checkpoint['model_state_dict'])
    model_loaded.to(device)
    
    # オプティマイザの再構築
    optimizer_loaded = torch.optim.Adam(model_loaded.parameters(), lr=3e-4)
    optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch_loaded = checkpoint.get('epoch', None)
    val_losses_loaded = checkpoint.get('val_losses', None)
    
    print(f"エポック {epoch_loaded} までのモデルをロードしました。")
    
    return model_loaded, optimizer_loaded, epoch_loaded, val_losses_loaded, test_dataloader

if __name__ == "__main__":
    
    save_dir = "transformer/models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "transformer_model.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_model, loaded_optimizer, loaded_epoch, loaded_val_losses, test_dataloader = load_model(save_path, device)
    
    # ロードしたモデルを使用してテスト
    test_acc = train.test_loop(test_dataloader, loaded_model, device)
    print(f"ロードしたモデルのテスト精度: {test_acc}%")