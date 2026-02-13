import CNN_BiLSTM.continuous_sign_language.modeling.functions as functions
import CNN_BiLSTM.models.cnn_bilstm_model as model
import CNN_BiLSTM.continuous_sign_language.modeling.config as model_config
import CNN_BiLSTM.continuous_sign_language.config as config
import CNN_BiLSTM.continuous_sign_language.dataset as dataset
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

# --- 1. モデルの準備 ---
mode = "test"
test_hdf5files, val_hdf5files, key2token = dataset.read_dataset(mode=mode)
# VOCABサイズ（Blank含む）の確定
VOCAB = len(key2token)

# モデルのインスタンス化 (model.Modelの引数にすべての設定を渡す)
# ここで model_instance が PyTorch の nn.Module になります
model_instance = model.Model(
    vocabulary=key2token,
    in_channels=196,  # 位置情報の入力次元
    hand_size=config.spatial_spatial_feature,  # 距離情報の入力次元
    cnn_out_channels=model_config.cnn_out_channels,
    cnn_dropout_rate=model_config.cnn_dropout_rate,
    conv_type=model_config.conv_type,
    use_bn=model_config.use_bn,
    kernel_sizes=model_config.kernel_sizes,
    num_layers=model_config.num_layers,
    num_heads=model_config.num_heads,
    dropout=model_config.dropout,
    num_classes=VOCAB,
    blank_id=0,
    cnn_model_type=model_config.cnn_model_type,
    temporal_model_type=model_config.temporal_model_type,
)

# 2. 学習済み重みのロード
MODEL_PATH = model_config.model_use_path
device = torch.device("cpu")

# 重みをロードする際は、必ず eval モードに設定します
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if "model_state_dict" in checkpoint:
        model_instance.load_state_dict(checkpoint["model_state_dict"])
        print("✅ model_state_dictから重みをロードしました")
    else:
        model_instance.load_state_dict(checkpoint)
        print("✅ 重みをロードしました")
    print(f"✅ モデルの重みをロードしました: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ 重みのロードに失敗しました（ファイルがない場合はスキップされます）: {e}")

model_instance.eval()


# --- 2.5. Core ML用のラッパーモデル作成 ---
class CoreMLWrapper(nn.Module):
    """Core ML変換用のラッパーモデル（推論専用、logitsのみ返す）"""

    def __init__(self, model):
        super().__init__()
        self.model = model
        # PackedSequenceを使わないようにする
        # BiLSTMのRNN層を直接アクセス可能にする
        if hasattr(model.temporal_model, "rnn"):
            self.rnn = model.temporal_model.rnn
        self.cnn_model = model.cnn_model
        self.spatial_correlation = model.spatial_correlation
        self.classifier = model.classifier
        self.cnn_model_type = model.cnn_model_type

    def forward(self, src_feature, spatial_feature, input_lengths):
        # 元のモデルのforward処理を手動で実行（PackedSequenceをスキップ）
        N, C, T, J = src_feature.shape
        src_feature_reshaped = (
            src_feature.permute(0, 3, 1, 2).contiguous().view(N, C * J, T)
        )
        spatial_feature_reshaped = spatial_feature.permute(0, 2, 1)

        # CNNモデルの実行
        if self.cnn_model_type == "DualCNNWithCTC":
            cnn_out, cnn_logit, updated_lgt = self.cnn_model(
                skeleton_feat=src_feature_reshaped,
                hand_feat=spatial_feature_reshaped,
                lgt=input_lengths,
            )
        else:
            cnn_out, cnn_logit, updated_lgt = self.cnn_model(
                skeleton_feat=src_feature_reshaped,
                spatial_feature=spatial_feature_reshaped,
                lgt=input_lengths,
            )

        # 相関学習モジュール（注意機構の可視化なし）
        cnn_out = self.spatial_correlation(cnn_out)

        # BiLSTM/Transformerの実行（PackedSequenceなし）
        # cnn_outは[T, B, C]形式
        # RNNを直接呼び出す（pack_padded_sequenceをスキップ）
        rnn_outputs, _ = self.rnn(cnn_out)  # PackedSequence不使用

        # 分類器の実行
        outputs = self.classifier(rnn_outputs)

        # logitsテンソルのみ返す（デコード結果は含まない）
        return outputs


wrapped_model = CoreMLWrapper(model_instance)
wrapped_model.eval()

# --- 3. Core ML 変換の設定 ---

# フレーム数（時間軸）を可変（1〜500フレーム）に設定
flexible_time = ct.RangeDim(lower_bound=1, upper_bound=500, default=150)

# モデルは [B, C, T, J] 形式を期待 (196 = C * J = 2 * 98)
# Swift側では [B, T, C*J] 形式のデータを用意し、
# モデル内で自動的に reshape されるようにする
inputs = [
    ct.TensorType(name="src_feature", shape=(1, 2, flexible_time, 98)),  # [B, C, T, J]
    ct.TensorType(
        name="spatial_feature", shape=(1, flexible_time, 24)
    ),  # [B, T, spatial_dim]
    ct.TensorType(name="input_lengths", shape=(1,), dtype=np.int32),  # [B]
]

# 4. TorchScriptへのトレース
print("🔧 TorchScriptにトレース中...")
# ダミー入力の作成
# モデルは [batch, C, T, J] 形式を期待 (196 = C * J = 2 * 98)
dummy_src_feature = torch.randn(1, 2, 150, 98)  # [B, C, T, J]
dummy_spatial_feature = torch.randn(1, 150, 24)  # [B, T, spatial_dim]
dummy_input_lengths = torch.tensor([150], dtype=torch.long)  # [B]

# ラッパーモデルをトレース（3つの引数のみ）
# strict=Falseで条件分岐の警告を抑制
traced_model = torch.jit.trace(
    wrapped_model,
    (dummy_src_feature, dummy_spatial_feature, dummy_input_lengths),
    strict=False,
)

# 5. Core ML への変換実行
print("🚀 PyTorchモデルをCore MLに変換中...")
mlmodel = ct.convert(
    traced_model,
    inputs=inputs,
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
)

# 6. モデルメタデータの付与
mlmodel.author = "Jun Shibata"
mlmodel.license = "Master's Thesis Project"
mlmodel.short_description = (
    "2-Stream CNN-BiLSTM for Sign Language Recognition (220-dim features)"
)

# 7. 保存
mlmodel.save("SignLanguageModel.mlpackage")
print("✨ 保存完了: SignLanguageModel.mlpackage")
