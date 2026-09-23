"""
Ті самі питання до трьох версій моделі — порівняти відповіді.

Запуск:
    python bench/quality_ollama.py
"""

import requests

URL = "http://localhost:11434"
MODELS = [
    "qwen2.5:0.5b-instruct-fp16",
    "qwen2.5:0.5b-instruct-q8_0",
    "qwen2.5:0.5b-instruct-q4_K_M",
]
QUESTIONS = [
    "What is 17 * 23? Answer with a number only.",
    "What is the capital of Australia? Answer in one word.",
    "Write a Python function that reverses a string.",
    "Explain gradient descent in one sentence.",
    "Поясни, що таке нейронна мережа, одним реченням.",
]


def ask(model, question):
    response = requests.post(
        f"{URL}/api/generate",
        json={
            "model": model,
            "prompt": question,
            "stream": False,
            "options": {"num_predict": 80, "temperature": 0, "seed": 0},
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


if __name__ == "__main__":
    for question in QUESTIONS:
        print("=" * 70)
        print("ПИТАННЯ:", question)
        for model in MODELS:
            tag = model.split("-")[-1]
            print(f"\n[{tag}]")
            print(ask(model, question))
    print("=" * 70)