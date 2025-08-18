import CNN_BiLSTM.continuous_sign_language.dataset as dataset
import CNN_BiLSTM.continuous_sign_language.modeling.functions as functions
import CNN_BiLSTM.models.cnn_bilstm_model as model
import CNN_BiLSTM.continuous_sign_language.modeling.config as model_config
import CNN_BiLSTM.continuous_sign_language.plots as plot
import CNN_BiLSTM.continuous_sign_language.init_log as init_log

import torch
import os
import numpy as np
import logging


def model_train():
    logger, log_file = init_log.setup_logging()
    logging.info("訓練を開始しました")
    train_hdf5files, val_hdf5files, test_hdf5files, key2token = dataset.read_dataset()
    train_dataloader, val_dataloader, test_dataloader, in_channels = (
        functions.set_dataloader(
            key2token, train_hdf5files, val_hdf5files, test_hdf5files
        )
    )

    VOCAB = len(key2token)
    out_channels = VOCAB
    pad_token = key2token["<pad>"]
    blank_id = 0
    print(key2token)
    print(
        f"VOCAB サイズ: {VOCAB}, パッディングトークン: {pad_token}, ブランクID: {blank_id}"
    )

    # モデルの初期化
    cnn_transformer = model.CNNBiLSTMModel(
        in_channels=in_channels,
        kernel_size=model_config.kernel_size,
        cnn_out_channels=model_config.cnn_out_channels,
        stride=model_config.stride,
        padding=model_config.padding,
        dropout_rate=model_config.dropout_rate,
        bias=model_config.bias,
        resNet=model_config.resNet,
        activation=model_config.activation,
        tren_num_layers=model_config.tren_num_layers,
        tren_num_heads=model_config.tren_num_heads,
        tren_dim_ffw=model_config.tren_dim_ffw,
        tren_dropout=model_config.tren_dropout,
        tren_norm_eps=model_config.tren_norm_eps,
        batch_first=model_config.batch_first,
        tren_norm_first=model_config.tren_norm_first,
        tren_add_bias=model_config.tren_add_bias,
        num_classes=out_channels,
        blank_idx=blank_id,
        temporal_model_type=model_config.temporal_model_type,  # 追加
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 過学習対策として適切な学習率とweight decayを設定
    optimizer = torch.optim.Adam(
        cnn_transformer.parameters(),
        lr=model_config.lr,  # 直接config.pyの学習率を使用
        weight_decay=(
            model_config.weight_decay if hasattr(model_config, "weight_decay") else 8e-5
        ),
    )

    # 学習率スケジューラ: 第2回の設定に戻す
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 最初の再開までのエポック数
        T_mult=2,  # 再開間隔の倍率
        eta_min=model_config.lr * 0.01,  # 最小学習率
    )
    
    epochs = model_config.epochs

    train_losses = []
    val_losses = []
    min_loss = float("inf")
    cnn_transformer.to(device)

    # plotsディレクトリを作成
    os.makedirs("plots", exist_ok=True)

    print(
        f"学習率: {model_config.lr:.6f}, Weight decay: {model_config.weight_decay if hasattr(model_config, 'weight_decay') else 1e-4:.6f}"
    )
    print("Start training.")
    for epoch in range(epochs):
        print("-" * 80)
        print(f"Epoch {epoch+1}")
        train_loss, train_times = functions.train_loop(
            dataloader=train_dataloader,
            model=cnn_transformer,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            return_pred_times=True,
        )
        train_losses.append(train_loss)

        if (epoch + 1) % model_config.eval_every_n_epochs == 0:
            val_loss, val_times = functions.val_loop(
                dataloader=val_dataloader,
                model=cnn_transformer,
                device=device,
                return_pred_times=True,
                current_epoch=epoch + 1,
            )
            val_losses.append(val_loss)

            if min_loss > val_loss:  # lossが最小なのを保存
                functions.save_model(
                    save_path=model_config.model_save_path,
                    model_default_dict=cnn_transformer.state_dict(),
                    optimizer_dict=optimizer.state_dict(),
                    epoch=model_config.epochs,
                )
                min_loss = val_loss

    train_losses_array = np.array(train_losses)
    val_losses_array = np.array(val_losses)
    print(
        f"Minimum validation loss:{val_losses_array.min()} at {np.argmin(val_losses_array)+1} epoch."
    )

    plot.train_loss_plot(train_losses_array)
    plot.val_loss_plot(val_losses_array, model_config.eval_every_n_epochs)
    plot.train_val_loss_plot(
        train_losses_array, val_losses_array, model_config.eval_every_n_epochs
    )



if __name__ == "__main__":
    model_train()
