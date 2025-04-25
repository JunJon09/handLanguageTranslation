activation="relu"

#detail 1DCNN
cnn_out_channels = 512
kernel_size =30
stride = 1
padding = kernel_size//2
dropout_rate = 0.2
bias = False
resNet = 0 # 0: なし, 1: restNet18, 2: restNet34 3: restNet50 restNet101 5: restNet152, 6: restNet152


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
epochs = 50
eval_every_n_epochs = 1

#モデルの保存
model_save_path = "one_dcnn_transformer_encoder/models/cnn_transformer_model.pth"
