import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)

model.eval()

messages = [
    {"role" : "system", "content": "Ypu are technical assistant, answer concisely"},
    {"role" : "user", "content": "Explain embeddings in two sentences"}
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize = False, add_generation_prompt = True
)

print(prompt)

inputs = tokenizer(prompt, return_tensors='pt')

start = time.time()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
elapsed = time.time() - start

new_tokens = output[0][inputs.input_ids.shape[-1]:]
answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(answer)
print(f"\n[{len(new_tokens)} токенів за {elapsed:.1f} с = "
      f"{len(new_tokens) / elapsed:.1f} tok/s]")
