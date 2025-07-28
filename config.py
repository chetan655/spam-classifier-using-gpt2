

cfg = {
    "emb_dim": 512,
    "vocab_size": 50257,
    "qkv_bias": True,
    "n_heads": 8,
    "context_length": 1024,
    "dropout": 0.1,
    "n_layers": 12,
    "preload": None,
    "weight_folder": "weights",
    "weight_basename": "_tmodel",
    "weight_decay": 0.1,
    "learning_rate": 5e-5,
    "epoch": 5,
    "eval_freq": 20,
    "eval_iter": 10,
    "num_classes": 2,
    "batch_size": 2,
    "num_workers": 0

}