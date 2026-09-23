import argparse

import requests

URL = "http://localhost:11434"
SHORT_PROMPT = "Explain what embeddings are in machine learning."
LONG_PROMPT = (
    "Machine learning is a field of study in artificial intelligence "
    "concerned with the development of statistical algorithms. "
) * 40 + "\nSummarize the text above in one sentence."
NEW_TOKENS = 64
RUNS = 3


def generate(model, prompt):
    """Один запит до Ollama. Повертає словник зі статистикою."""
    response = requests.post(
        f"{URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": NEW_TOKENS,
                "temperature": 0,
                "seed": 0,
                "num_ctx": 4096,
            },
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def loaded_size_gb(model):
    """Скільки памʼяті займає модель усередині Ollama."""
    info = requests.get(f"{URL}/api/ps", timeout=10).json()
    for m in info.get("models", []):
        if m.get("name") == model or m.get("model") == model:
            return m["size"] / 1e9
    return float("nan")


def bench(model, prompt):
    generate(model, "warm up")  # розігрів: завантажує модель у памʼять

    ns = 1e9  # Ollama віддає час у наносекундах
    ttfts, prefill_speeds, decode_speeds = [], [], []

    for i in range(RUNS):
        # унікальний початок, щоб Ollama не взяла промпт із кешу
        stats = generate(model, f"[run {i}] {prompt}")

        prefill_s = stats["prompt_eval_duration"] / ns
        decode_s = stats["eval_duration"] / ns

        ttfts.append(prefill_s)
        prefill_speeds.append(stats["prompt_eval_count"] / prefill_s)
        decode_speeds.append(stats["eval_count"] / decode_s)
        prompt_tokens = stats["prompt_eval_count"]

    mean = lambda xs: sum(xs) / len(xs)
    return {
        "model": model,
        "size_gb": loaded_size_gb(model),
        "prompt_tokens": prompt_tokens,
        "ttft_ms": 1000 * mean(ttfts),
        "prefill_tok_s": mean(prefill_speeds),
        "decode_tok_s": mean(decode_speeds),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--long", action="store_true")
    args = parser.parse_args()

    prompt = LONG_PROMPT if args.long else SHORT_PROMPT
    r = bench(args.model, prompt)

    print(f"\n{'model':32s} {'памʼять ГБ':>10s} {'prompt':>6s} "
          f"{'TTFT мс':>8s} {'prefill tok/s':>13s} {'decode tok/s':>12s}")
    print(f"{r['model']:32s} {r['size_gb']:10.2f} {r['prompt_tokens']:6d} "
          f"{r['ttft_ms']:8.0f} {r['prefill_tok_s']:13.0f} {r['decode_tok_s']:12.1f}")