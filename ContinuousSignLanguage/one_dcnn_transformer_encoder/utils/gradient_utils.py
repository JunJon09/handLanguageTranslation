import torch
import logging

logger = logging.getLogger(__name__)

def check_gradients(model, name="model"):
    """
    モデルの勾配をチェックし、NaNや無限大の値を検出して報告します。
    
    Args:
        model: PyTorchモデル
        name: モデルを識別するための名前（ログ用）
    
    Returns:
        bool: 勾配が有効（NaNや無限大が含まれていない）ならTrue
    """
    total_params = 0
    bad_params = 0
    
    for tag, param in model.named_parameters():
        if param.grad is not None:
            total_params += 1
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                bad_params += 1
                logger.warning(f"{name}の勾配に問題が見つかりました: {tag}")
                
                # 勾配の統計情報を記録
                if param.grad.numel() > 0:
                    with torch.no_grad():
                        grad_min = param.grad.min().item()
                        grad_max = param.grad.max().item()
                        grad_mean = param.grad.mean().item()
                        grad_std = param.grad.std().item()
                        nan_count = torch.isnan(param.grad).sum().item()
                        inf_count = torch.isinf(param.grad).sum().item()
                        
                        logger.warning(f"  勾配の統計: min={grad_min}, max={grad_max}, mean={grad_mean}, std={grad_std}")
                        logger.warning(f"  問題のある値: NaN={nan_count}, Inf={inf_count}")
    
    if bad_params > 0:
        logger.warning(f"{name}の勾配に問題があります: {bad_params}/{total_params}のパラメータに不正値があります")
        return False
    
    return True

def clip_gradients(model, max_norm=1.0, name="model"):
    """
    モデルの勾配をクリッピングし、NaNや無限大の値を0に置き換えます。
    
    Args:
        model: PyTorchモデル
        max_norm: 勾配のL2ノルムの最大値
        name: モデルを識別するための名前（ログ用）
    
    Returns:
        bool: 勾配が有効（NaN/Infを0に置き換えた後）ならTrue
    """
    # まず全パラメータの勾配のNaNや無限大をチェック
    has_any_bad = False
    
    for tag, param in model.named_parameters():
        if param.grad is not None:
            bad_grads = torch.isnan(param.grad) | torch.isinf(param.grad)
            if bad_grads.any():
                has_any_bad = True
                # NaNや無限大の値を0に置き換え
                param.grad[bad_grads] = 0.0
                logger.warning(f"{name}の勾配を修正しました: {tag}のNaN/Inf値を0に置換")
    
    # グローバルなノルムクリッピング
    if has_any_bad:
        logger.warning(f"{name}の勾配に問題があったため修正しました")
    
    # 勾配クリッピングを適用
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    return not has_any_bad

def safe_optimizer_step(optimizer, model, max_norm=1.0, name="model"):
    """
    安全なオプティマイザのステップ実行
    - 勾配をチェック
    - 問題のある場合は修正（NaN/Infを0に）
    - 勾配クリッピング
    - オプティマイザステップ

    Args:
        optimizer: PyTorchオプティマイザ
        model: PyTorchモデル
        max_norm: 勾配のL2ノルムの最大値
        name: モデルを識別するための名前（ログ用）
    
    Returns:
        bool: 勾配の更新が成功したか（すべて有効だったか）
    """
    # 勾配のチェック（ロギングのみ）
    check_gradients(model, name)
    
    # 勾配のクリッピングと修正
    clip_result = clip_gradients(model, max_norm, name)
    
    # オプティマイザステップ実行
    optimizer.step()
    
    return clip_result
