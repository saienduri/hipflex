# hipflex Demo

Run any GPU workload with a hard VRAM limit and optional compute restriction — no code changes, no driver mods. Just `LD_PRELOAD`.

hipflex intercepts HIP APIs at the library level and enforces per-process VRAM limits and CU restrictions transparently. Frameworks like PyTorch, vLLM, and SGLang see your configured limit as the GPU's total memory and CU count. An MI325X with 256 GiB / 304 CUs becomes a 24 GiB / 38 CU GPU — and every framework adapts automatically.

## Setup

```bash
# Download the latest release
wget -q https://github.com/saienduri/hipflex/releases/download/v0.1.0/libhipflex.so

# That's it. No install, no dependencies, no root.
```

## Quick Start

```bash
# Limit any GPU process to 24 GiB of VRAM
LD_PRELOAD=./libhipflex.so FH_MEMORY_LIMIT=24GiB python3 your_script.py

# Restrict to 38 Compute Units (no memory limit)
LD_PRELOAD=./libhipflex.so FH_CU_RANGE=0-37 python3 your_script.py

# Limit memory AND restrict CUs
LD_PRELOAD=./libhipflex.so FH_MEMORY_LIMIT=24GiB FH_CU_RANGE=0-37 python3 your_script.py
```

---

## Demo 1: PyTorch + Transformers — Mistral-7B-Instruct-v0.3

Load and run inference with Mistral 7B inside a 24 GiB VRAM budget.

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /proc:/host/proc:ro \
  -v $PWD/libhipflex.so:/workspace/libhipflex.so:ro \
  -e LD_PRELOAD=/workspace/libhipflex.so \
  -e FH_MEMORY_LIMIT=24GiB \
  -e HIP_VISIBLE_DEVICES=0 \
  rocm/pytorch:latest \
  bash -c '
    pip install -q transformers accelerate
    python3 << "EOF"
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=== Before model load ===")
free, total = torch.cuda.mem_get_info(0)
print(f"GPU:        {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {total / 1024**3:.1f} GiB")
print(f"Used VRAM:  {(total - free) / 1024**3:.1f} GiB")

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float16, device_map="auto"
)

print("\n=== After model load ===")
free, total = torch.cuda.mem_get_info(0)
print(f"Total VRAM: {total / 1024**3:.1f} GiB")
print(f"Used VRAM:  {(total - free) / 1024**3:.1f} GiB")

messages = [{"role": "user", "content": "Write a haiku about GPU memory."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
t0 = time.time()
output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
elapsed = time.time() - t0
tokens = output.shape[1] - inputs.input_ids.shape[1]
print(f"\n=== Generation ===")
print(tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
print(f"\n{tokens} tokens in {elapsed:.1f}s ({tokens/elapsed:.1f} tok/s)")

free, total = torch.cuda.mem_get_info(0)
print(f"\n=== Final memory ===")
print(f"Used: {(total - free) / 1024**3:.1f} GiB / {total / 1024**3:.1f} GiB limit")
EOF
  '
```

**What you'll see** (tested on MI325X, physical VRAM 256 GiB):

```
=== Before model load ===
GPU:        AMD Instinct MI325X
Total VRAM: 24.0 GiB                     <-- hipflex limit, not 256 GiB
Used VRAM:  0.0 GiB

=== After model load ===
Total VRAM: 24.0 GiB
Used VRAM:  13.5 GiB                     <-- 7B model in FP16

=== Generation ===
Silicon dance,
Pixels swirl in endless stream,
Memory's precious gem.

23 tokens in 3.8s (6.0 tok/s)

=== Final memory ===
Used: 13.6 GiB / 24.0 GiB limit         <-- 10.4 GiB still available
```

PyTorch's caching allocator sees 24 GiB total and manages within it. The 7B FP16 model (13.5 GiB) fits comfortably with room for KV cache and activations. On exit, hipflex drains all tracked allocations back to the shared memory counter.

---

## Demo 2: vLLM — Gemma 4 E4B

Serve Google's Gemma 4 E4B (4B effective parameters, multimodal, 140 languages) with vLLM's OpenAI-compatible API. vLLM automatically sizes its KV cache to fit within the hipflex limit.

```bash
# Start the server
docker run -d --name vllm-demo \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /proc:/host/proc:ro \
  -v $PWD/libhipflex.so:/workspace/libhipflex.so:ro \
  -e LD_PRELOAD=/workspace/libhipflex.so \
  -e FH_MEMORY_LIMIT=24GiB \
  -e HIP_VISIBLE_DEVICES=0 \
  -p 8000:8000 \
  rocm/pytorch:latest \
  bash -c '
    pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/ -q
    pip install --upgrade transformers -q
    vllm serve google/gemma-4-E4B-it \
      --host 0.0.0.0 --port 8000 \
      --attention-backend TRITON_ATTN \
      --gpu-memory-utilization 0.85 \
      --max-model-len 4096
  '

# Wait for startup (~120s for pip install + model load + CUDA graph compilation)
until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 5; done

# Query the API
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E4B-it",
    "messages": [{"role": "user", "content": "Explain what LD_PRELOAD does in Linux in 2 sentences."}],
    "max_tokens": 150
  }' | python3 -m json.tool

# Cleanup
docker rm -f vllm-demo
```

**What you'll see** (tested on MI325X, physical VRAM 256 GiB):

```
vLLM server logs:
  Model loading took 16.38 GiB memory     <-- Gemma 4 E4B weights
  Available KV cache memory: 2.74 GiB     <-- vLLM auto-sized to fit 24 GiB limit
  GPU KV cache size: 29,968 tokens

Memory state:
  Total VRAM: 24.0 GiB (hipflex limit)
  Used VRAM:  ~20.4 GiB                   <-- model weights + KV cache + CUDA graphs
  Free VRAM:  ~3.6 GiB

Response:
  `LD_PRELOAD` is an environment variable in Linux that allows a user to
  specify a list of shared libraries that should be loaded before any other
  libraries, including the standard ones. This mechanism is often used for
  debugging, patching, or replacing specific functions in target programs
  without modifying their binaries.

  23 prompt + 60 completion = 83 tokens
```

vLLM probes available GPU memory at startup and fits its KV cache into whatever hipflex reports. With 24 GiB, the 16.4 GiB model leaves 2.7 GiB for KV cache (~30K tokens) after accounting for CUDA graphs and runtime overhead. Gemma 4 requires `--attention-backend TRITON_ATTN` for its bidirectional image-token attention. On a full 256 GiB GPU without hipflex, the same model would have vastly more KV cache — hipflex makes vLLM behave as if the hardware is smaller.

---

## Demo 3: SGLang — Qwen2.5-7B-Instruct

Serve Qwen 2.5 7B with SGLang's high-throughput engine using AMD's aiter attention backend.

```bash
# Start the server
docker run -d --name sglang-demo \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --shm-size 32g \
  -v /proc:/host/proc:ro \
  -v $PWD/libhipflex.so:/workspace/libhipflex.so:ro \
  -e LD_PRELOAD=/workspace/libhipflex.so \
  -e FH_MEMORY_LIMIT=24GiB \
  -e HIP_VISIBLE_DEVICES=0 \
  -e SGLANG_USE_AITER=1 \
  -p 30000:30000 \
  lmsysorg/sglang:v0.5.9-rocm700-mi30x \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 30000 \
    --tp 1 \
    --mem-fraction-static 0.8 \
    --attention-backend aiter

# Wait for startup (~60s)
until curl -s http://localhost:30000/health > /dev/null 2>&1; do sleep 5; done

# Query the API
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Compare GPU memory management to a parking garage in 3 sentences."}],
    "max_tokens": 150
  }' | python3 -m json.tool

# Cleanup
docker rm -f sglang-demo
```

**What you'll see** (tested on MI325X):

```
SGLang server logs:
  max_total_num_tokens=37784              <-- token budget sized to 24 GiB limit
  available_gpu_mem=3.02 GB               <-- SGLang sees remaining memory after model load
  KV cache dtype: torch.bfloat16

Memory state:
  Total VRAM: 24.0 GiB (hipflex limit)
  Used VRAM:  21.0 GiB                   <-- model + KV cache pool
  Free VRAM:   3.0 GiB

Response:
  Imagine a GPU as a parking garage where each memory block is like a parking
  space. Just as a parking garage efficiently manages space to park cars, a GPU
  manages its memory to store and quickly access data for processing tasks. The
  system must also ensure that no space is wasted and that data can be retrieved
  as quickly as possible, much like how a well-managed parking garage ensures
  smooth traffic flow and minimal empty spaces.

  42 prompt + 84 completion = 126 tokens
```

SGLang reads `hipMemGetInfo` at startup to determine its token budget. With 24 GiB, it fits ~37K tokens of KV cache in BF16 alongside the 7B model. The aiter attention backend runs AMD-optimized kernels — all within the hipflex-enforced limit.

---

## The Key Insight

Every framework queries GPU capabilities the same way: `hipMemGetInfo`, `hipDeviceTotalMem`, `hipGetDeviceProperties`. hipflex intercepts all of these and reports your configured limits. The frameworks never know they're on a fractionalized GPU — they just see a smaller one and adapt.

| Metric | Without hipflex | With hipflex (24 GiB, 38 CUs) |
|--------|----------------|-------------------------------|
| `torch.cuda.mem_get_info()` total | 256.0 GiB | 24.0 GiB |
| `hipGetDeviceProperties` multiProcessorCount | 304 | 38 |
| `rocm-smi --showmeminfo vram` total | 256 GiB | 24 GiB |
| `amd-smi static` VRAM size | 256 GiB | 24 GiB |
| vLLM KV cache allocation | ~200 GiB | 2.7 GiB |
| SGLang token budget | ~400K tokens | 37K tokens |
| Allocation beyond limit | Succeeds | `hipErrorOutOfMemory` |
| Compute Units available | 304 | 38 (hardware-enforced via `HSA_CU_MASK`) |

## How It Works

hipflex uses `LD_PRELOAD` to intercept 27 HIP memory APIs (`hipMalloc`, `hipFree`, `hipMemGetInfo`, etc.) before they reach the GPU driver:

1. **Enforces memory limits** — allocations that would exceed `FH_MEMORY_LIMIT` are denied with `hipErrorOutOfMemory`
2. **Restricts compute** — `FH_CU_RANGE` sets `HSA_CU_MASK` before the HIP runtime initializes, hardware-limiting which Compute Units the process can use
3. **Spoofs device info** — `hipMemGetInfo`, `hipDeviceTotalMem`, `hipGetDeviceProperties` (including `multiProcessorCount`), `rocm-smi`, and `amd-smi` all report the configured limits
4. **Tracks overhead** — kernel-level VRAM usage (page tables, code objects, scratch buffers) is tracked via KFD sysfs and subtracted from the budget
5. **Drains on exit** — when the process exits, hipflex returns all tracked allocations to the shared memory counter

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FH_MEMORY_LIMIT` | *(none)* | VRAM limit per GPU (e.g., `24GiB`, `8G`, `4096M`) |
| `FH_CU_RANGE` | *(none)* | CU restriction, applied uniformly to all visible GPUs (e.g., `0-37` for 38 CUs). Sets `HSA_CU_MASK` and spoofs `multiProcessorCount`. Works independently or with `FH_MEMORY_LIMIT` |
| `FH_LOG_PATH` | *(none)* | Log output: `stderr`, or a file path |
| `FH_HIP_LIB_PATH` | `libamdhip64.so` | Path to HIP runtime library |
| `FH_ENABLE_HOOKS` | `true` | Set to `false` to disable all hooks |

## Accounting Validation

hipflex tracks memory at two levels: explicit allocations via `hipMalloc` interception, and implicit kernel-level overhead via KFD sysfs (`/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>`). Every 100th allocation triggers a reconciliation that cross-checks the two. Here are real numbers from the demos above (MI325X, 24 GiB limit):

### PyTorch — Mistral-7B-Instruct-v0.3

| Metric | Value |
|--------|-------|
| Tracked allocations at exit | 6 allocs, 13931 MiB |
| `torch.cuda.mem_get_info()` used | 13.6 GiB (~13926 MiB) |
| SHM drain on exit | 13931 → 0 MiB (clean) |
| Reconciliation events | 1 (first alloc only — PyTorch's caching allocator does few large hipMalloc calls) |

### vLLM — Gemma 4 E4B

| Reconciliation | KFD VRAM | hipMalloc tracked | Kernel overhead | Effective limit | Delta |
|---|---|---|---|---|---|
| alloc 0 (init) | 2 MiB | 2 MiB | 0 MiB | 24576 MiB | — |
| alloc 100 | 11042 MiB | 10509 MiB | 532 MiB | 24043 MiB | 1 MiB |
| alloc 200 | 15492 MiB | 14963 MiB | 532 MiB | 24043 MiB | 3 MiB |
| alloc 300 | 21439 MiB | 19798 MiB | 1640 MiB | 22935 MiB | 1 MiB |
| alloc 400 | 21633 MiB | 19804 MiB | 1828 MiB | 22747 MiB | 1 MiB |

3 processes initialized (main → registry subprocess → worker fork). Dead process reaping confirmed.

### SGLang — Qwen2.5-7B-Instruct

| Reconciliation | KFD VRAM | hipMalloc tracked | Kernel overhead | Effective limit | Delta |
|---|---|---|---|---|---|
| alloc 0 (init) | 2 MiB | 2 MiB | 0 MiB | 24576 MiB | — |
| alloc 100 | 12367 MiB | 11849 MiB | 517 MiB | 24058 MiB | 1 MiB |
| alloc 200 | 22857 MiB | 21518 MiB | 1324 MiB | 23251 MiB | 15 MiB |
| alloc 300 | 22831 MiB | 21499 MiB | 1332 MiB | 23243 MiB | 0 MiB |

5 processes total (launcher, 2 spawn workers, 2 torchinductor JIT subprocesses). Dead process reaping confirmed.

### What the columns mean

- **KFD VRAM**: actual per-process GPU memory footprint from `/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>` (ground truth)
- **hipMalloc tracked**: sum of all live allocations intercepted by hipflex
- **Kernel overhead**: `KFD VRAM - hipMalloc tracked` — memory allocated by the kernel directly (page tables, compiled GPU kernels, scratch buffers, HSA state) that bypasses `hipMalloc`
- **Effective limit**: `mem_limit - total_overhead` — the dynamically tightened enforcement ceiling
- **Delta**: `(KFD VRAM - hipMalloc tracked) - reported overhead` — accounting error (0 = exact match)

No `ACCOUNTING DRIFT` or `HIGH OVERHEAD` warnings fired for any framework. The max delta across all reconciliations was 15 MiB (0.06% of the 24 GiB limit), caused by in-flight allocations between the KFD sysfs read and the tracker snapshot.

## Compatibility

- **GPUs**: AMD Instinct MI250X, MI300X, MI325X, MI350x, MI355x (any ROCm-supported GPU)
- **ROCm**: 6.x, 7.x (TheRock releases included)
- **Frameworks**: PyTorch, vLLM, SGLang, any HIP application
- **No code changes required** — works with any existing binary or container
