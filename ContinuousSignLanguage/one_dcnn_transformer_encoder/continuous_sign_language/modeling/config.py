inter_channels = 512
activation="relu"

#detail 1DCNN
kernel_size = 5
stride = 1
padding = 1

#detail transformer encoder
tren_num_layers=6
tren_num_heads=8
tren_dim_ffw=256
tren_dropout_pe=0.1
tren_dropout=0.1
norm_type = "layer"
tren_norm_eps=1e-5
tren_norm_first=True
tren_add_bias=True
tren_add_tailnorm=True
batch_first = True

#学習パラメータ
label_smoothing = 0.1
lr = 3e-4
epochs = 100
eval_every_n_epochs = 1
max_seqlen = 150

#モデルの保存
model_save_dir = "cnn_transformer/models"
model_save_path = "cnn_transformer_model.pth"