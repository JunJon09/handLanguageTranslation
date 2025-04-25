import one_dcnn_transformer_encoder.continuous_sign_language.dataset as dataset
import one_dcnn_transformer_encoder.continuous_sign_language.modeling.functions as functions
import one_dcnn_transformer_encoder.models.one_dcnn_transformer_encoder as model
import one_dcnn_transformer_encoder.continuous_sign_language.modeling.config as model_config
import one_dcnn_transformer_encoder.continuous_sign_language.plots as plot
import torch
import os
import numpy as np


def model_train():
    train_hdf5files, val_hdf5files, test_hdf5files, key2token = dataset.read_dataset()
    train_dataloader, val_dataloader, test_dataloader, in_channels = (
        functions.set_dataloader(
            key2token, train_hdf5files, val_hdf5files, test_hdf5files
        )
    )

    VOCAB = len(key2token)
    out_channels = VOCAB
    pad_token = key2token["<pad>"]
    blank_id = VOCAB - 1
    print(key2token)
    print(
        f"VOCAB サイズ: {VOCAB}, パッディングトークン: {pad_token}, ブランクID: {blank_id}"
    )

    # モデルの初期化
    cnn_transformer = model.OnedCNNTransformerEncoderModel(
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
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # より低い学習率で開始し、学習の安定性を向上
    initial_lr = model_config.lr * 0.1  # 元の学習率の1/10
    optimizer = torch.optim.Adam(cnn_transformer.parameters(), lr=initial_lr)

    # 学習率スケジューラを追加
    # ウォームアップ後に徐々に学習率を下げるスケジューラ
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=model_config.lr,  # 最大学習率は元の設定値
        steps_per_epoch=len(train_dataloader),
        epochs=model_config.epochs,
        pct_start=0.3,  # 学習率がピークに達するまでのエポック割合
        div_factor=10.0,  # 初期学習率 = max_lr / div_factor
        final_div_factor=100.0,  # 最終学習率 = max_lr / (div_factor * final_div_factor)
    )

    epochs = model_config.epochs

    train_losses = []
    val_losses = []
    min_loss = float("inf")
    cnn_transformer.to(device)

    # plotsディレクトリを作成
    os.makedirs("plots", exist_ok=True)

    print(f"初期学習率: {initial_lr:.6f}, 最大学習率: {model_config.lr:.6f}")
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
