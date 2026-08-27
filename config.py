MODEL_CONFIGS={
    "tiny":{"num_heads":4,
            "num_hiddens":64,
            "query_size":64,
            "key_size":64,
            "value_size":64,
            "d_ffn":128,
            "num_layers":2,
            "dropout":0.1},

    "base":{"num_heads":8,
            "num_hiddens":512,
            "query_size":512,
            "key_size":512,
            "value_size":512,
            "d_ffn":2048,
            "num_layers":6,
            "dropout":0.1},

    "big":{"num_heads":16,
           "num_hiddens":1024,
           "query_size":1024,
           "key_size":1024,
           "value_size":1024,
           "d_ffn":4096,
           "num_layers":6,
           "dropout":0.3}}

TRAINING_CONFIGS={
    "tiny":{"max_steps":550,
            "warmup_steps":100,
            "max_examples":1000,
            "valid_max_examples":500,
            "max_tokens":128,
            "buffer_size":1000,
            "pool_size":200,
            "token_budget":512,
            "log_every_steps":100,
            "save_every_steps":100},

    "base":{"max_steps":100_000,
            "warmup_steps":4_000,
            "max_examples":None,
            "valid_max_examples":None,
            "max_tokens":256,
            "buffer_size":10_000,
            "pool_size":1000,
            "token_budget":None,
            "log_every_steps":1000,
            "save_every_steps":1000},

    "big":{"max_steps":300_000,
           "warmup_steps":4_000,
           "max_examples":None,
           "valid_max_examples":None,
           "max_tokens":256,
           "buffer_size":10_000,
           "pool_size":1000,
           "token_budget":None,
           "log_every_steps":1000,
           "save_every_steps":1000}}

def get_training_config(model_size):
    return TRAINING_CONFIGS[model_size].copy()

def get_model_config(model_size,vocab_size):
    model_config=MODEL_CONFIGS[model_size].copy()
    model_config["vocab_size"]=vocab_size
    return model_config