# Docker環境でのimportパスを修正
try:
    from CNN_BiLSTM.continuous_sign_language import config
except ImportError:
    from . import config
