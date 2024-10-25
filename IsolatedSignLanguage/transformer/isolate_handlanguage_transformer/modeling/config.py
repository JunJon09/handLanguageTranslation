inter_channels = 64
activation = "relu"
tren_num_layers = 6
tren_num_heads = 2
tren_dim_ffw = 256
tren_dropout_pe = 0.1
tren_dropout = 0.1
tren_layer_norm_eps = 1e-5
lr = 3e-4
epochs = 50
eval_every_n_epochs = 1
model_save_dir = "transformer/models"
model_save_path = "transformer_model.pth"
tren_norm_first = False
tren_add_bias = True
tren_add_tailnorm = False