import CNN_BiLSTM.continuous_sign_language.modeling.functions as functions
import CNN_BiLSTM.models.cnn_bilstm_model as model
import CNN_BiLSTM.continuous_sign_language.modeling.config as model_config
import CNN_BiLSTM.continuous_sign_language.config as config
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

    cnn_transformer = model.Model(
        vocabulary=key2token,
        in_channels=in_channels,
        hand_size=config.spatial_spatial_feature,
        cnn_out_channels=model_config.cnn_out_channels,
        cnn_dropout_rate=model_config.cnn_dropout_rate,
        conv_type=model_config.conv_type,
        use_bn=model_config.use_bn,
        kernel_sizes=model_config.kernel_sizes,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        dropout=model_config.dropout,
        num_classes=out_channels,
        blank_id=0, # CTCのblankインデックスを0に設定
        cnn_model_type=model_config.cnn_model_type,
        temporal_model_type=model_config.temporal_model_type,
    )

    load_model, optimizer_loaded, epoch_loaded = functions.load_model(
        cnn_transformer, save_path, device
    )

    wer, test_times = functions.test_loop(
        dataloader=test_dataloader,
        model=load_model,
        device=device,
        return_pred_times=True
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
