# llm-playground

Hands-on benchmarks of LLM inference: how serving engine, weight format and
concurrency affect latency, throughput and memory.

All numbers below were measured, not copied from vendor pages. Every claim in
this README comes from a script in `bench/`.

**Hardware**

- CPU: AVX2, no hardware `bf16` support
- GPU: RTX 3090, 24 GB (rented on vast.ai)

**Models**: Qwen2.5-0.5B-Instruct, Qwen2.5-7B-Instruct
**Workloads**: short prompt (~40 tokens) and long prompt (~765 tokens), 64 new tokens

---

## TL;DR

| Change | Effect |
|---|---|
| transformers → Ollama (CPU, same fp16 weights) | decode ×2.8, prefill ×14 |
| fp16 → Q4_K_M quantization (CPU) | decode ×1.8, **prefill −15%** |
| CPU → GPU (transformers, 0.5B) | decode ×4.6, prefill ×4.7 |
| transformers → vLLM with batching (GPU, 7B) | total throughput **×71** |

The single biggest win was not quantization and not even moving to a GPU — it
was switching to an engine actually designed for inference.

---

## Background: two phases, two bottlenecks

**Prefill** processes the whole prompt in one forward pass. Measured as TTFT.
It is *compute-bound*: weights are read once and applied to all prompt tokens at
the same time, so the processor is busy doing arithmetic.

**Decode** generates one token per pass. Measured as tokens/s.
It is *memory-bound*: every single token requires reading all model weights, and
the processor spends most of its time waiting for data.

This gives a simple estimate:

```
decode tok/s ≈ memory bandwidth / model size in bytes
```

Measured on CPU — three different configurations converge on the same effective
bandwidth, which is a property of the machine, not of the model:

| format | size | decode tok/s | size × speed |
|---|---|---|---|
| fp16 | 1.08 GB | 31.9 | 34 GB/s |
| Q8_0 | 0.62 GB | 45.2 | 28 GB/s |
| Q4_K_M | 0.48 GB | 58.1 | 28 GB/s |

The practical consequence of the split: **a single change can speed up one phase
and slow down the other**, so the two must always be measured separately.

---

## Results: CPU

### transformers, Qwen2.5-0.5B

| dtype | prompt | weights | TTFT ms | prefill tok/s | decode tok/s |
|---|---|---|---|---|---|
| fp32 | 42 | 1.98 GB | 575 | 159 | 5.5 |
| bf16 | 42 | 0.99 GB | 1459 | 27 | 11.5 |
| fp32 | 760 | 1.98 GB | 4774 | 159 | 5.9 |
| bf16 | 760 | 0.99 GB | 27700 | 27 | 9.2 |

Halving the weights doubled decode speed, exactly as the memory-bound formula
predicts. But TTFT got **2.5× worse** on the short prompt and **5.5× worse** on
the long one.

Cause: this CPU has AVX2 but no `avx512_bf16`, so every bf16 operation is
converted to fp32 and back. Decode barely notices — the processor was idle
waiting for memory anyway. Prefill pays the full price, because it is
compute-bound.

Total time for a 64-token answer:

```
short prompt:  fp32 12.0 s   bf16  6.9 s   → bf16 wins
long prompt:   fp32 16.1 s   bf16 33.7 s   → fp32 wins, by 2×
```

There is no universally better dtype — only a better one for a given CPU and a
given prompt profile.

### Ollama / llama.cpp, Qwen2.5-0.5B

| format | file size | memory | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| fp16 | 994 MB | 1.08 GB | 378 | 31.9 |
| Q8_0 | 531 MB | 0.62 GB | 344 | 45.2 |
| Q4_K_M | 397 MB | 0.48 GB | 320 | 58.1 |

Same model, same precision (fp16), same machine as the transformers run above —
**decode ×2.8 and prefill ×14 faster**. llama.cpp ships hand-written kernels for
AVX2 and does not round-trip through fp32.

Quantization then adds decode speed but costs prefill: 4-bit weights have to be
unpacked before multiplication, which is extra compute in the phase that is
already compute-bound.

That reverses the ranking on long prompts (total time for 64 tokens):

| format | short prompt | long prompt (765 tok) |
|---|---|---|
| fp16 | 2.1 s | 6.7 s |
| Q8_0 | **1.5 s** | **5.9 s** |
| Q4_K_M | **1.2 s** | 8.0 s |

**For a RAG workload on CPU, Q8_0 is the right choice — not the smallest quant.**

### Compression ratios

Q4_K_M is only 2.50× smaller than fp16, not 4×:

- one scale factor stored per 32 weights
- `K_M` is a mixed-precision scheme: some tensors stay at 6 bits
- the embedding table is 136M of 494M parameters in this model and is quantized
  conservatively

On larger models the embedding share is much smaller, so the ratio gets closer
to 3.5–4×.

### Quality cost

Five greedy questions across fp16 / Q8_0 / Q4_K_M:

- Q8_0 is effectively indistinguishable from fp16 — 3 of 5 answers were
  identical word for word
- Q4_K_M rewords answers and produced one factual slip
- all three failed the same factual question ("capital of Australia" → Sydney)

That last point matters: **the shared failure is a model limitation, not a
quantization artifact.** Always check full precision before blaming the quant.

---

## Results: GPU

### transformers, RTX 3090

| model | dtype | weights | TTFT ms | prefill tok/s | decode tok/s |
|---|---|---|---|---|---|
| 0.5B | bf16 | 1.00 GB | 21.2 | 1790 | 53.0 |
| 0.5B | fp16 | 1.00 GB | 20.8 | 1827 | 54.9 |
| 0.5B | fp32 | 1.98 GB | 22.7 | 1674 | 53.1 |
| 7B | bf16 | 15.28 GB | 27.0 | 1409 | 44.7 |

The bf16 prefill penalty seen on CPU disappears — the 3090 has hardware support
for 16-bit formats.

But decode is suspiciously slow. The 3090 has ~936 GB/s of memory bandwidth, so
a 1 GB model should reach hundreds of tokens/s. Two clues point at the real
bottleneck:

1. fp32 and bf16 give the **same** decode speed, although one is twice the size
2. a **15× larger model is only 16% slower**

```
0.5B  (1.00 GB):  53.0 tok/s → 18.9 ms per step
7B   (15.28 GB):  44.7 tok/s → 22.4 ms per step
```

The 3.5 ms difference is the actual GPU work on weights. The remaining ~19 ms is
fixed Python/PyTorch dispatch overhead — hundreds of CUDA kernel launches per
step, independent of model size. **The GPU is idle roughly 85% of the time.**

This is why `transformers` is a training library, not a serving stack.

### vLLM, Qwen2.5-7B, saturation curve

Short prompt (40 tokens), 64 new tokens:

| concurrency | latency s | total tok/s | per user tok/s |
|---|---|---|---|
| 1 | 1.26 | 50.6 | 50.6 |
| 4 | 1.30 | 196.2 | 49.0 |
| 16 | 1.40 | 714.0 | 44.6 |
| 64 | 1.61 | 2440.0 | 38.1 |
| 128 | 2.48 | **3164.5** | 24.7 |
| 256 | 5.19 | 3031.0 | 11.8 |
| 512 | 7.77 | 3029.9 | 5.9 |
| 1024 | 13.12 | 2966.6 | 2.9 |

Up to ~128 concurrent requests, extra load is nearly free: it fills the idle GPU
time measured above. Weights are read once per step regardless of batch size, so
32 users cost roughly the same memory traffic as one.

Past saturation, throughput flatlines at ~3000 tok/s while latency grows
linearly — additional requests just queue.

```
transformers, 1 request:   44.7 tok/s
vLLM, 128 requests:      3164.5 tok/s     ×71 on identical hardware
```

**Production operating range: 64–128 concurrent requests, queue beyond that.**

### Long prompts hit a different wall

Prompt 763 tokens:

| concurrency | fp16 cache: time / tok/s | fp8 cache: time / tok/s |
|---|---|---|
| 32 | — | 5.89 / 101.1 |
| 64 | 10.24 / 150.1 | 5.96 / 198.9 |
| 128 | 11.69 / 262.9 | 11.27 / 283.1 |

Ceiling drops from ~3164 to ~283 tok/s — an order of magnitude. The limit here
is not compute but KV-cache capacity and prefill time.

`--kv-cache-dtype fp8` doubles cache capacity: at 64 concurrent requests
eviction stopped and wall time halved. At 128 it barely helps, because the
bottleneck moved to prefill (98k prompt tokens to process).

**Implication for RAG: shortening the prompt beats tuning the engine.** A
reranker that keeps 3 documents instead of 10 directly multiplies service
throughput.

---

## Memory planning

KV-cache per token:

```
2 (K and V) × layers × kv_heads × head_dim × bytes_per_value
```

Qwen2.5-7B: `2 × 28 × 4 × 128 × 2 = 56 KB/token`.

| scenario | weights | cache | total |
|---|---|---|---|
| 7B fp16, 1 user, 4096 ctx | 15.2 | 0.23 | 15.4 GB |
| 7B fp16, 16 users | 15.2 | 3.76 | 19.0 GB |
| same, hypothetically without GQA | 15.2 | 26.3 | 41.5 GB |
| 7B Q4, 30 users, 8192 ctx | 4.5 | 14.0 | 18.5 GB |

Two things worth noting. **Grouped Query Attention (4 kv-heads instead of 28)
shrinks the cache 7×** — without it this model would not serve multiple users on
a single 24 GB card. And **under load the cache, not the weights, becomes the
dominant consumer**, which is why quantizing weights alone stops helping at some
point.

`bench/memory_estimate.py` predicted 15.2 GB of weights; vLLM reported 15.28 GB
— under 0.5% error. But that estimate is a lower bound: CUDA context,
activations and fragmentation added ~1.7 GB in practice, so plan with 10–20%
headroom.

fp32 for a 7B model does not fit in 24 GB at all: 30.4 GB of weights → CUDA OOM.

---

## Practical rules derived from these runs

1. Do not serve with `transformers` — Ollama is several times faster on CPU,
   vLLM is tens of times faster on GPU.
2. On CPU for RAG, pick Q8_0 rather than the smallest quant: prefill dominates.
3. On GPU, cap concurrency at the saturation point and queue the rest.
4. Enable `--kv-cache-dtype fp8` for long-context workloads — nearly free.
5. Compute memory before renting: weights + cache, plus 10–20% headroom.
6. Benchmark the regime you will actually run. A break-even point extrapolated
   from short prompts did not hold for long ones.

## Methodology notes

- Prefill and decode measured separately; one change can help one and hurt the other.
- Warm-up pass before every measurement — the first call is always slower.
- `torch.cuda.synchronize()` before and after timing on GPU, otherwise you
  measure Python dispatch instead of the GPU.
- Prefix caching distorts repeated identical prompts — a unique prefix is
  prepended per run.
- Run-to-run spread on CPU is 1–9%; smaller differences are noise.
- Thread-per-request load generation breaks around 1024 concurrent connections
  (`can't start new thread`); asyncio or k6 is needed beyond that.

---

## Repository layout

```
bench/
  bench_transformers.py   # HuggingFace transformers, CPU and GPU
  bench_ollama.py         # Ollama HTTP API, GGUF quantization
  bench_vllm.py           # vLLM OpenAI-compatible endpoint, concurrency sweep
  memory_estimate.py      # weights + KV-cache estimate from config.json alone
  quality_ollama.py       # side-by-side answers across quantization levels
learn/
  ...                     # PyTorch fundamentals, attention from scratch
EXPERIMENTS.md            # full lab notebook with raw numbers
```

## Running the benchmarks

```bash
# CPU, transformers
python bench/bench_transformers.py --dtype bf16
python bench/bench_transformers.py --dtype fp32 --long

# CPU, Ollama (server must be running)
ollama pull qwen2.5:0.5b-instruct-q8_0
python bench/bench_ollama.py --model qwen2.5:0.5b-instruct-q8_0

# GPU, vLLM
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --kv-cache-dtype fp8

python bench/bench_vllm.py --concurrency 64
python bench/bench_vllm.py --concurrency 64 --long

# memory estimate, no download required
python bench/memory_estimate.py
```