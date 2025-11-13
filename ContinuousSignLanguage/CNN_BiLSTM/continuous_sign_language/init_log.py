import logging
import os
from datetime import datetime, timezone, timedelta

def setup_logging(log_dir="./CNN_BiLSTM/logs", log_level=logging.INFO, console_output=False):
    """
    ログの設定を行う関数（日本時間対応・pytz不要版）
    
    Args:
        log_dir: ログファイルの保存ディレクトリ
        log_level: ログレベル
        console_output: Trueの場合ターミナルにも出力、Falseの場合ファイルのみ
    """
    # ログディレクトリの作成
    os.makedirs(log_dir, exist_ok=True)
    
    # 日本時間（UTC+9）のタイムゾーンを設定
    jst = timezone(timedelta(hours=9))
    
    # 日本時間でのタイムスタンプ付きログファイル名
    timestamp = datetime.now(jst).strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 日本時間用のカスタムフォーマッター
    class JSTFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            # レコードの時刻を日本時間に変換
            dt = datetime.fromtimestamp(record.created, tz=jst)
            if datefmt:
                return dt.strftime(datefmt)
            else:
                return dt.strftime('%Y-%m-%d %H:%M:%S JST')
    
    # ログフォーマットの設定（日本時間対応）
    formatter = JSTFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S JST'
    )
    
    # ルートロガーの設定
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 既存のハンドラーをクリア
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # ファイルハンドラーの設定
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # コンソールハンドラーの設定（オプション）
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 最初のメッセージはprintで表示（ファイル作成の確認のため）
    print(f"ログファイルを作成しました: {log_file}")
    logger.info(f"ログ設定完了 - ファイル出力: {log_file}, コンソール出力: {console_output}")
    
    return logger, log_file