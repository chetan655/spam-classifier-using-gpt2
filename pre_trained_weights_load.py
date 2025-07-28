import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# from model import Model
# from config import cfg
from gpt_download3 import download_and_load_gpt2



def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    
    if not torch.is_tensor(right):
        right = torch.tensor(right, dtype=left.dtype, device=left.device)
    else:
        right = right.to(device=left.device, dtype=left.dtype)
    return torch.nn.Parameter(right)


def load_weights_into_gpt(gpt, params):

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):

        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf[b].mha.q_w.weight = assign(
            gpt.trf[b].mha.q_w.weight, q_w.T
        )
        gpt.trf[b].mha.v_w.weight = assign(
            gpt.trf[b].mha.v_w.weight, v_w.T
        )
        gpt.trf[b].mha.k_w.weight = assign(
            gpt.trf[b].mha.k_w.weight, k_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf[b].mha.q_w.bias = assign(
            gpt.trf[b].mha.q_w.bias, q_b
        )
        gpt.trf[b].mha.v_w.bias = assign(
            gpt.trf[b].mha.v_w.bias, v_b
        )
        gpt.trf[b].mha.k_w.bias = assign(
            gpt.trf[b].mha.k_w.bias, k_b
        )

        gpt.trf[b].mha.out.weight = assign(
            gpt.trf[b].mha.out.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf[b].mha.out.bias = assign(
            gpt.trf[b].mha.out.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        gpt.trf[b].ff.layer1.weight = assign(
            gpt.trf[b].ff.layer1.weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.trf[b].ff.layer1.bias = assign(
            gpt.trf[b].ff.layer1.bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf[b].ff.layer2.weight = assign(
            gpt.trf[b].ff.layer2.weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.trf[b].ff.layer2.bias = assign(
            gpt.trf[b].ff.layer2.bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        gpt.trf[b].norm1.scale = assign(
            gpt.trf[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf[b].norm1.shift = assign(
            gpt.trf[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf[b].norm2.scale = assign(
            gpt.trf[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf[b].norm2.shift = assign(
            gpt.trf[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )

    gpt.norm.scale = assign(gpt.norm.scale, params["g"])
    gpt.norm.shift = assign(gpt.norm.shift, params["b"])
    # gpt.proj_out.weight = assign(gpt.proj_out.weight, params["wte"])


def load(cfg, Model):
    
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
       "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
       "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    model_name = "gpt2-small (124M)"
    new_config = cfg.copy()
    new_config.update(model_configs[model_name])
    new_config.update({"context_length": 1024, "qkv_bias": True})

    gpt = Model(new_config)
    gpt.eval()

    print("Model loader with new_config, named as gpt.")
    print(f"Model config: {new_config}")

    print("Downloading pre-trained weights...")
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    print("Weights downloaded!")

    print("Loading weights to model(gpt)...")
    load_weights_into_gpt(params=params, gpt=gpt)
    print("Pre-trained weights loaded to model(gpt)!")


    for param in gpt.parameters():
        param.requires_grad = False      # freezing all weigths 
  
    for param in gpt.trf[-1].parameters():
        param.requires_grad = True

    for param in gpt.norm.parameters():
        param.requires_grad = True

    return gpt, new_config