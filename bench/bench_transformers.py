import gc
import time
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = ("Machine learning is a field of study in artificial intelligence "
          "concerned with the development of statistical algorithms. ") * 40 \
         + "\nSummarize the text above in one sentence."
NEW_TOKENS = 64
RUNS = 3

def rss_gb():
    return psutil.Process().memory_info().rss / 1e9

def time_generate(model, inputs, n_tokens, pad_id):
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = n_tokens,
            min_new_tokens = n_tokens,
            do_sample = False,
            pad_token_id = pad_id,
        )
    elapsed = time.perf_counter() - start
    n_new = out.shape[-1] - inputs.input_ids.shape[-1]
    return elapsed, n_new

def bench(model_name, dtype):
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype = dtype)
    model.eval()
    load_s = time.perf_counter() - t0

    messages = [{"role" : "user", "content" : PROMPT}]
    text = tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = True
    )

    inputs = tokenizer(text, return_tensors = "pt")
    n_prompt = inputs.input_ids.shape[-1]
    pad_id = tokenizer.eos_token_id

    time_generate(model, inputs, 4, pad_id)

    ttfts, speeds = [], []
    for _ in range(RUNS):
        ttft, _ = time_generate(model, inputs, 1, pad_id)
        total, n_new = time_generate(model, inputs, NEW_TOKENS, pad_id)
        decode_speed = (n_new - 1) / (total - ttft)
        ttfts.append(ttft)
        speeds.append(decode_speed)

    result = {
        "model": model_name.split("/")[-1],
        "dtype": str(dtype).replace("torch.", ""),
        "load_s": load_s,
        "weights_gb": model.get_memory_footprint() / 1e9,
        "rss_gb": rss_gb(),
        "prompt_tokens": n_prompt,
        "ttft_ms": 1000 * sum(ttfts) / len(ttfts),
        "tok_s": sum(speeds) / len(speeds),
    }

    del model
    gc.collect()
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    r = bench(args.model, dtype)

    print(f"\n{'model':28s} {'dtype':9s} {'prompt':>6s} {'ваги ГБ':>8s} "
          f"{'RSS ГБ':>7s} {'TTFT мс':>8s} {'tok/s':>6s} {'prefill tok/s':>13s}")
    print(f"{r['model']:28s} {r['dtype']:9s} {r['prompt_tokens']:6d} "
          f"{r['weights_gb']:8.2f} {r['rss_gb']:7.2f} {r['ttft_ms']:8.0f} "
          f"{r['tok_s']:6.1f} {r['prompt_tokens'] / (r['ttft_ms'] / 1000):13.0f}")
