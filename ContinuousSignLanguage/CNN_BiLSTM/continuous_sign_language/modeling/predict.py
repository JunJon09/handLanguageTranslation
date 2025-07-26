import CNN_BiLSTM.continuous_sign_language.modeling.functions as functions
import CNN_BiLSTM.models.cnn_bilstm_model as model
import CNN_BiLSTM.continuous_sign_language.modeling.config as model_config
import CNN_BiLSTM.continuous_sign_language.dataset as dataset
import CNN_BiLSTM.continuous_sign_language.init_log as init_log
import torch
import logging

if __name__ == "__main__":
    logger, log_file = init_log.setup_logging()
    logging.info("テストを開始しました")
    train_hdf5files, val_hdf5files, test_hdf5files, key2token = dataset.read_dataset()
    train_dataloader, val_dataloader, test_dataloader, in_channels = (
        functions.set_dataloader(
            key2token, train_hdf5files, val_hdf5files, test_hdf5files
        )
    )
    VOCAB = len(key2token)
    out_channels = VOCAB
    save_path = model_config.model_save_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        blank_idx=VOCAB - 1,
        temporal_model_type=model_config.temporal_model_type,  # 追加
    )

    load_model, optimizer_loaded, epoch_loaded = functions.load_model(
        cnn_transformer, save_path, device
    )

    # ========================================
    # 🔍 可視化・分析設定
    # ========================================
    VISUALIZE_ATTENTION = True  # True: 可視化する, False: 可視化しない
    GENERATE_CONFUSION_MATRIX = True  # True: 混同行列を生成, False: 生成しない
    VISUALIZE_CONFIDENCE = True  # True: 予測信頼度可視化, False: 可視化しない
    VISUALIZE_MULTILAYER_FEATURES = True  # True: 多層特徴量可視化, False: 可視化しない
    MULTILAYER_METHOD = "both"  # "tsne", "umap", "both"

    if (
        VISUALIZE_ATTENTION
        or GENERATE_CONFUSION_MATRIX
        or VISUALIZE_CONFIDENCE
        or VISUALIZE_MULTILAYER_FEATURES
    ):
        analysis_options = []
        if VISUALIZE_ATTENTION:
            analysis_options.append("Attention & CTC可視化")
        if GENERATE_CONFUSION_MATRIX:
            analysis_options.append("混同行列分析")
        if VISUALIZE_CONFIDENCE:
            analysis_options.append("予測信頼度可視化")
        if VISUALIZE_MULTILAYER_FEATURES:
            analysis_options.append(f"多層特徴量可視化({MULTILAYER_METHOD})")

        print(f"🔍 拡張分析モードでテストを実行します")
        print(f"  有効な分析: {', '.join(analysis_options)}")
        print(
            f"  多層特徴量分析: CNN空間パターン、BiLSTM時系列、Attention重要度、最終統合特徴量"
        )

        wer, test_times = functions.test_loop(
            dataloader=test_dataloader,
            model=load_model,
            device=device,
            return_pred_times=True,
            blank_id=VOCAB - 1,
            visualize_attention=VISUALIZE_ATTENTION,
            generate_confusion_matrix=GENERATE_CONFUSION_MATRIX,
            visualize_confidence=VISUALIZE_CONFIDENCE,
            visualize_multilayer_features=VISUALIZE_MULTILAYER_FEATURES,
            multilayer_method=MULTILAYER_METHOD,
        )
    else:
        print("📊 通常モードでテストを実行します")
        wer, test_times = functions.test_loop(
            dataloader=test_dataloader,
            model=load_model,
            device=device,
            return_pred_times=True,
            blank_id=VOCAB - 1,
            visualize_attention=False,
            generate_confusion_matrix=False,
            visualize_confidence=False,
            visualize_multilayer_features=False,
            multilayer_method="both",
        )

    print(f"ロードしたモデルのテスト精度: {wer}")
