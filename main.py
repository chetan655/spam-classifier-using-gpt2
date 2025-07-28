# from pre_trained_weights_load import load_weights_into_gpt
from dataset import train_loader, val_loader
from train import train
from model import gpt, new_config

# from model import Model
# from pre_trained_weights_load import load
# # from gpt_download3 import download_and_load_gpt2
# from config import cfg

# Model = Model

# gpt, new_config = load(cfg=cfg, Model=Model)


# model_configs = {
#     "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
#     "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
#     "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
#     "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
# }

# model_name = "gpt2-small (124M)"
# new_config = cfg.copy()
# new_config.update(model_configs[model_name])
# new_config.update({"context_length": 1024, "qkv_bias": True})

# gpt = Model(new_config)
# gpt.eval()

# print("Model loader with new_config, named as gpt.")
# print(f"Model config: {new_config}")

# print("Downloading pre-trained weights...")
# settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
# print("Weights downloaded!")

# print("Loading weights to model(gpt)...")
# load_weights_into_gpt(params=params, gpt=gpt)
# print("Pre-trained weights loaded to model(gpt)!")


# for param in gpt.parameters():
#     param.requires_grad = False      # freezing all weigths 
  
# for param in gpt.trf[-1].parameters():
#     param.requires_grad = True

# for param in gpt.norm.parameters():
#     param.requires_grad = True







train(new_config=new_config, model=gpt, train_loader=train_loader, val_loader=val_loader, eval_freq=new_config["eval_freq"], eval_iter=new_config["eval_iter"])