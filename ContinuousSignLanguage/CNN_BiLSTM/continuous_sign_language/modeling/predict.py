import CNN_BiLSTM.continuous_sign_language.modeling.functions as functions
import CNN_BiLSTM.models.cnn_bilstm_model as model
import CNN_BiLSTM.continuous_sign_language.modeling.config as model_config
import CNN_BiLSTM.continuous_sign_language.dataset as dataset
import CNN_BiLSTM.continuous_sign_language.init_log as init_log
import CNN_BiLSTM.continuous_sign_language.modeling.performance_monitor as pm
import torch
import logging


if __name__ == "__main__":
    mode = "test"
    init_log.setup_logging(mode=mode)
    
    # パフォーマンス監視初期化
    monitor = pm.PerformanceMonitor(monitor_interval=1.0)
    
    logging.info("🚀 テストを開始しました")
    
    # パフォーマンス監視開始
    monitor.start_monitoring()
    
    test_hdf5files, val_hdf5files, key2token = dataset.read_dataset(mode=mode)
    test_dataloader, val_dataloader, in_channels = functions.set_dataloader(key2token, test_hdf5files, val_hdf5files, mode)
    print(f"🔢 テストデータ数: {len(test_dataloader.dataset)} サンプル")
    VOCAB = len(key2token)
    out_channels = VOCAB
    save_path = model_config.model_use_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cnn_transformer = model.CNNBiLSTMModel(
        vocabulary=key2token,
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

    # Transformerファインチューニングモードの設定確認
    if (
        hasattr(model_config, "fine_tune_transformer_only")
        and model_config.fine_tune_transformer_only
    ):
        if model_config.temporal_model_type in [
            "transformer",
            "multiscale_transformer",
        ]:
            logging.info(
                "🎯 予測時：Transformerファインチューニングモードで訓練されたモデルを使用"
            )
        # 予測時はフリーズ設定は不要（全層を使用して予測）

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
    
        # パフォーマンス監視停止と結果表示
    monitor.stop_monitoring()
    monitor.print_summary()
    
    # 詳細なリソース使用状況をログに記録
    summary = monitor.get_summary()
    if isinstance(summary, dict):
        logging.info("=" * 50)
        logging.info("パフォーマンス監視結果詳細")
        logging.info("=" * 50)
        logging.info(f"監視時間: {summary['monitoring_duration']:.1f}秒")
        logging.info(f"CPU使用率 - 平均: {summary['cpu_usage']['avg']:.1f}%, 最大: {summary['cpu_usage']['max']:.1f}%, 最小: {summary['cpu_usage']['min']:.1f}%")
        logging.info(f"メモリ使用量 - 平均: {summary['memory_usage_mb']['avg']:.0f}MB, 最大: {summary['memory_usage_mb']['max']:.0f}MB, 最小: {summary['memory_usage_mb']['min']:.0f}MB")
        if torch.cuda.is_available():
            logging.info(f"GPU メモリ - 平均: {summary['gpu_memory_mb']['avg']:.0f}MB, 最大: {summary['gpu_memory_mb']['max']:.0f}MB, 最小: {summary['gpu_memory_mb']['min']:.0f}MB")
        else:
            logging.info("GPU: 利用不可")
        logging.info("=" * 50)
    
    logging.info("✅ テストが完了しました")
    
   
        # パフォーマンス監視が確実に停止されるようにする
    if 'monitor' in locals() and monitor.is_monitoring:
            monitor.stop_monitoring()
