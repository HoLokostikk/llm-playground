from transformers import AutoConfig


def estimate(model_name, params_billion, bytes_per_param=2,
             context=4096, batch=1):
    cfg = AutoConfig.from_pretrained(model_name)

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    layers = cfg.num_hidden_layers
    kv_heads = cfg.num_key_value_heads

    weights_gb = params_billion * 1e9 * bytes_per_param / 1e9

    kv_per_token = 2 * layers * kv_heads * head_dim * bytes_per_param
    kv_gb = kv_per_token * context * batch / 1e9

    # те саме, якби GQA не було: kv-голів стільки ж, скільки q-голів
    kv_no_gqa_gb = kv_gb * cfg.num_attention_heads / kv_heads

    print(f"\n{model_name}")
    print(f"  шари={layers}  q-голови={cfg.num_attention_heads}  "
          f"kv-голови={kv_heads}  head_dim={head_dim}")
    print(f"  ваги:                       {weights_gb:6.2f} ГБ")
    print(f"  KV-кеш на 1 токен:          {kv_per_token / 1024:6.1f} КБ")
    print(f"  KV-кеш {context} ток × {batch}:      {kv_gb:6.2f} ГБ")
    print(f"  KV-кеш без GQA:             {kv_no_gqa_gb:6.2f} ГБ")
    print(f"  разом:                      {weights_gb + kv_gb:6.2f} ГБ")


if __name__ == "__main__":
    estimate("Qwen/Qwen2.5-0.5B-Instruct", params_billion=0.49)
    estimate("Qwen/Qwen2.5-7B-Instruct", params_billion=7.6)
    estimate("Qwen/Qwen2.5-7B-Instruct", params_billion=7.6, batch=16)