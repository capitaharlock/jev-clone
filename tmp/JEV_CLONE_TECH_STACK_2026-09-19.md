# JevClone — Technical Stack
## Local-first inference on Apple Silicon and CUDA, with a direct path to cloud deployment

**Date:** 19 September 2026  
**Project:** `jev-clone`

---

# 1. Core principle

JevClone should be **Rust-first**, but not Rust-only at the cost of slowing model research.

The production/runtime path is:

```text
TRAIN / RESEARCH
Python + PyTorch
        ↓
   SafeTensors
        ↓
RUST MODEL + RUNTIME
        ↓
CPU / CUDA / Metal
        ↓
Local binary or cloud API
```

Python is allowed in the training laboratory. Python is **not required in production inference**.

Training speed is dominated by GPU kernels rather than Python itself. PyTorch launches optimized native GPU kernels, so rewriting the training control loop in Rust does not automatically make tensor math faster. For inference, serving, startup, concurrency and deployment, however, Rust is the preferred base.

---

# 2. Languages

## Primary language: Rust

Use Rust for:

- model inference;
- runtime;
- local CLI;
- local daemon/server;
- public API;
- state cache;
- batching;
- request scheduler;
- production observability;
- cloud container;
- benchmark tooling;
- eventual custom CPU/CUDA/Metal operations.

## Secondary language: Python

Use Python only for:

- architecture research;
- training;
- dataset conversion;
- distillation;
- calibration experiments;
- reference implementations;
- model export.

## Kernel languages

Only when profiling proves they are necessary:

```text
CUDA C++ / CUDA kernels
Metal Shading Language
```

---

# 3. Repository layout

```text
jev-clone/
├── crates/
│   ├── jev-core/
│   ├── jev-model/
│   ├── jev-runtime/
│   ├── jev-server/
│   ├── jev-cli/
│   ├── jev-bench/
│   └── jev-format/
├── training/
│   ├── python/
│   ├── configs/
│   ├── datasets/
│   └── export/
├── kernels/
│   ├── cuda/
│   └── metal/
├── models/
│   ├── configs/
│   └── manifests/
├── docker/
│   ├── train/
│   └── inference/
├── benchmarks/
├── docs/
└── tests/
```

---

# 4. Rust ML framework

## Primary: Candle

Use Hugging Face Candle as the first Rust tensor/model framework:

https://github.com/huggingface/candle

Reasons:

- native Rust;
- CPU backend;
- CUDA backend;
- Metal backend;
- training support;
- custom operations/kernels;
- SafeTensors integration;
- transformer implementations;
- suitable for the three target environments.

Conceptually:

```text
Apple Silicon → Candle + Metal
NVIDIA        → Candle + CUDA
CPU           → Candle CPU
```

## Secondary candidate: Burn

Evaluate Burn only if Candle blocks us materially.

Do not maintain the same model in two Rust frameworks during early development.

Decision:

```text
Candle first.
Burn only as fallback or benchmark.
```

---

# 5. Canonical model artifact

Use:

```text
SafeTensors
```

Source:

https://github.com/huggingface/safetensors

Canonical release:

```text
model.safetensors
config.json
tokenizer.json
calibration.json
manifest.json
LICENSE
```

Do not make `.pt`/pickle the canonical release artifact.

The Rust runtime must reconstruct the model from configuration + tokenizer + SafeTensors without importing Python code.

---

# 6. Tokenizer

Use Hugging Face Tokenizers.

Rust runtime:

```text
tokenizers crate
```

Training:

```text
Python bindings / Transformers tokenizer
```

Pin tokenizer hash/version in the model manifest.

---

# 7. Training stack

Primary training framework:

```text
Python
PyTorch
```

Support packages:

```text
Transformers where useful
datasets
tokenizers
safetensors
accelerate where useful
```

Prefer a relatively small custom training loop rather than a large opaque training framework.

Maintain two representations of the architecture:

```text
Reference/training → PyTorch
Production        → Rust/Candle
```

Every release candidate needs numerical parity tests between them.

---

# 8. Apple Silicon: M4 Max / M5

Target machines:

```text
Mac M4 Max — 48 GB unified memory
Mac M5 — 48 GB unified memory
```

## Training

Start with:

```text
PyTorch MPS
```

Use the Macs for:

- development;
- fine-tuning;
- long-context experiments;
- model evaluation;
- latency tests;
- experiments that benefit from unified memory.

## Production inference

Primary:

```text
Rust + Candle + Metal
```

Alternative benchmark paths:

```text
MLX
ONNX Runtime + CoreML
CPU/Accelerate
```

MLX is useful for Apple-specific performance comparison, but should not become the project-wide runtime because it would split the architecture into an Apple-only path.

MLX:

https://github.com/ml-explore/mlx

---

# 9. Windows NVIDIA machine

Target:

```text
Windows
NVIDIA RTX 40xx/50xx-class GPU
CUDA
```

Exact GPU and VRAM must be detected before fixing memory budgets.

## Training

Primary:

```text
PyTorch + CUDA
```

## Production/local inference

Primary:

```text
Rust + Candle CUDA
```

Optimized alternatives:

```text
ONNX Runtime CUDA
TensorRT
```

This lets the Rust service keep one API while the execution backend changes underneath.

---

# 10. ONNX

ONNX is an optional deployment artifact, not the source of truth.

Canonical:

```text
SafeTensors + architecture config
```

Derived:

```text
model.onnx
TensorRT engine
CoreML artifact
```

Use ONNX for:

- CUDA benchmarks;
- TensorRT;
- CoreML execution;
- graph optimization.

Dynamic question/option counts may reduce graph optimization. If necessary, support shape buckets:

```text
options:   2 / 4 / 8 / 16 / 32 / 64
questions: 1 / 4 / 8 / 16 / 32
```

---

# 11. CUDA optimization order

Use this order:

```text
1. Candle CUDA baseline
2. ONNX Runtime CUDA
3. TensorRT
4. custom CUDA kernels
```

Only write custom CUDA after profiling.

For a small model, launch overhead, memory movement, state reuse and batching may matter as much as raw compute.

---

# 12. Metal optimization order

```text
1. Candle Metal
2. compare with PyTorch MPS
3. compare with MLX
4. ONNX/CoreML
5. custom Metal kernels if justified
```

The public API and model semantics must remain identical across backends.

---

# 13. CPU support

CPU remains a supported fallback:

```text
Apple CPU
x86-64
ARM64
```

Use:

```text
Candle CPU
Accelerate on macOS where useful
```

Small 20M–70M variants may be practical on CPU.

CPU support is also important for CI and development.

---

# 14. Precision

Support:

```text
FP32
FP16
BF16
INT8
```

Later:

```text
INT4
```

Training:

```text
BF16 where supported
FP16 otherwise
FP32 as reference
```

Production target:

```text
INT8 when quality/calibration remains acceptable
```

Recalibrate after quantization.

---

# 15. First local product

Produce one native executable:

```bash
jevclone serve   --model ./models/jevclone-v0.1   --device auto   --port 8080
```

Automatic device:

```text
Mac + Metal available        → Metal
NVIDIA + CUDA available      → CUDA
otherwise                    → CPU
```

Allow explicit override:

```bash
--device metal
--device cuda
--device cpu
```

---

# 16. Rust model API

The architecture must expose state reuse explicitly:

```rust
let state = engine.encode_state(input)?;
let decisions = engine.decide(&state, questions)?;
```

Separating:

```text
encode_state()
```

from:

```text
decide()
```

is fundamental because shared-state inference is one of JevClone's main performance advantages.

---

# 17. Local/cloud HTTP API

Use:

```text
Axum
Tokio
Serde
Tower
tracing
```

Start with HTTP/JSON.

Network latency dominates JSON parsing in Internet deployment, so do not prematurely require gRPC.

Optional later:

```text
gRPC
MessagePack
Unix sockets
```

---

# 18. Runtime concurrency

Do not allow every HTTP request to independently launch uncontrolled GPU work.

Architecture:

```text
Tokio request handlers
        ↓
request queue
        ↓
batch scheduler
        ↓
GPU worker
        ↓
response fan-out
```

This also enables dynamic batching.

---

# 19. Dynamic batching

Cloud runtime should optionally combine compatible requests arriving within a very short window.

Example:

```text
0–2 ms batching window
```

Group by:

```text
model
state-length bucket
question-count bucket
option-count bucket
precision
```

Latency-critical requests can bypass batching.

---

# 20. State cache

Rust service owns the state cache.

Key:

```text
model_version
state_hash
tokenizer_hash
```

Value:

```text
encoded state memory
```

Possible tiers:

```text
L1 GPU memory
L2 host RAM
```

Do not begin by storing live tensor memories in Redis. Keep state memory close to the worker/GPU that generated it.

---

# 21. Cloud prototype

Initial commercial deployment:

```text
Internet
   ↓
RunPod Serverless
   ↓
Rust container
   ↓
Candle CUDA / ONNX Runtime / TensorRT
   ↓
JevClone model
```

RunPod Serverless:

https://www.runpod.io/product/serverless

This fits the initial product because it supports:

- custom containers;
- autoscaling;
- scale-to-zero;
- per-use GPU economics;
- multiple regions later.

---

# 22. Production container

Inference image should contain:

```text
jevclone-server
required CUDA/runtime libraries
tokenizer
weights
calibration
configuration
```

Avoid shipping:

```text
Python
PyTorch training environment
datasets
Jupyter
development toolchain
```

Use a multi-stage Docker build.

---

# 23. Training container

Separate image:

```text
Python
PyTorch
CUDA
training dependencies
```

Structure:

```text
docker/
├── training/Dockerfile
└── inference/Dockerfile
```

Never deploy the training image as the public inference API.

---

# 24. Cloud training

When local hardware is insufficient, use:

```text
RunPod Pods
```

rather than Serverless inference workers.

Workflow:

```text
commit code
↓
build/pull training image
↓
start temporary GPU pod
↓
pull dataset shards
↓
train
↓
upload SafeTensors + metrics
↓
destroy pod
```

Because the training environment is Docker + CUDA, it remains portable to other GPU providers.

---

# 25. Cloud portability

Do not couple model/runtime code to RunPod.

Use a thin provider adapter:

```text
RunPod request envelope
        ↓
jev-server/core API
```

The same binary/container should later run on:

```text
RunPod
AWS
GCP
Azure
CoreWeave
Vast
bare metal
Kubernetes
```

---

# 26. Future multi-region architecture

Start:

```text
one US region
scale-to-zero
```

Later:

```text
US East
US West
Europe
```

Global routing:

```text
Cloudflare or equivalent
        ↓
nearest healthy region
        ↓
regional GPU endpoint
```

Do not design a decentralized compute network before traffic justifies it.

---

# 27. Cold-start optimization

A Rust-first inference image helps.

Optimize by:

- avoiding Python;
- keeping the binary compact;
- using compact SafeTensors;
- baking weights into the image when practical;
- otherwise using regional cached storage;
- loading tokenizer once;
- loading calibration once;
- preallocating reusable buffers;
- minimizing runtime dependencies.

A 100M–250M model should be dramatically easier to cold-start than a normal multi-billion-parameter generative LLM.

---

# 28. Observability

Use:

```text
tracing
OpenTelemetry
Prometheus-compatible metrics
```

Record:

```text
request latency
queue latency
tokenization time
state encode time
decision time
GPU utilization
GPU memory
cold starts
batch size
state-cache hit rate
errors
model version
```

Do not log user state content by default.

---

# 29. CI

GitHub Actions:

```text
cargo fmt
cargo clippy
cargo test
Python tests
PyTorch ↔ Candle parity tests
model format tests
ONNX export test
```

GPU-specific CI can be separate.

CPU CI must always verify model correctness.

---

# 30. Numerical parity

A model is not releasable until PyTorch reference and Rust runtime agree within defined tolerance.

Test:

```text
short state
long state
2 options
32 options
1 question
many questions
choice
boolean
score
```

Test separately for:

```text
FP32
FP16/BF16
INT8
```

where applicable.

---

# 31. Benchmark stack

Rust benchmark harness:

```text
criterion
custom latency runner
```

Measure:

```text
cold process startup
model load
tokenization
state encode
question fusion
decision head
total latency
```

Hardware:

```text
M4 Max
M5
Windows RTX
RunPod GPU
```

Report:

```text
p50
p95
p99
memory
throughput
```

---

# 32. Development phases

## Phase A — architecture research

```text
Python/PyTorch
SafeTensors
Rust model skeleton
```

Goal: prove architecture.

## Phase B — local product

```text
Rust
Candle
Metal
CUDA
CPU
Axum
CLI
```

Goal: same model runs locally on all three machines.

## Phase C — optimized local runtime

```text
INT8
ONNX/TensorRT benchmarks
Metal optimization
state cache
dynamic batching
```

## Phase D — cloud prototype

```text
Docker
RunPod Serverless
one US region
scale-to-zero
```

## Phase E — commercial deployment

```text
multi-region
global routing
warm capacity only if justified
production metrics
autoscaling
```

---

# 33. Technology matrix

| Component | Technology |
|---|---|
| Core language | Rust |
| Training language | Python |
| Research/training | PyTorch |
| Rust ML runtime | Candle |
| Apple GPU | Candle Metal |
| Windows/NVIDIA GPU | Candle CUDA |
| Optimized CUDA alternative | ONNX Runtime / TensorRT |
| Apple comparison path | MLX / CoreML |
| CPU | Candle CPU / Accelerate |
| Weights | SafeTensors |
| Tokenizer | Hugging Face Tokenizers |
| API | Axum + Tokio |
| Serialization | Serde / JSON |
| Logging | tracing |
| Metrics | OpenTelemetry / Prometheus |
| CLI | Rust |
| Containers | Docker |
| Cloud inference | RunPod Serverless |
| Cloud training | RunPod Pods |
| CI | GitHub Actions |

---

# 34. Machine matrix

| Machine | Training | Inference | Preferred backend |
|---|---|---|---|
| Mac M4 Max 48 GB | PyTorch MPS | Rust/Candle | Metal |
| Mac M5 48 GB | PyTorch MPS | Rust/Candle | Metal |
| Windows RTX | PyTorch CUDA | Rust/Candle | CUDA |
| RunPod training GPU | PyTorch CUDA | — | CUDA |
| RunPod inference GPU | — | Rust service | CUDA/TensorRT |
| CPU-only machine | limited | Rust/Candle | CPU |

---

# 35. Backend abstraction

Application logic must not care which accelerator runs the model.

Conceptual interface:

```rust
trait Backend {
    fn encode_state(&self, input: &Tokens) -> Result<StateMemory>;

    fn decide(
        &self,
        state: &StateMemory,
        questions: &[Question],
    ) -> Result<Vec<Decision>>;
}
```

Implementations:

```text
CandleCpuBackend
CandleMetalBackend
CandleCudaBackend
OnnxBackend
TensorRtBackend
```

Backend-independent:

```text
question schema
option semantics
calibration
decision types
state-cache contract
API responses
model configuration
```

Backend-specific:

```text
tensor allocation
kernel execution
CUDA
Metal
ONNX
TensorRT
```

---

# 36. Model artifact contract

```text
jevclone-v0.1/
├── config.json
├── tokenizer.json
├── model.safetensors
├── calibration.json
├── manifest.json
└── LICENSE
```

`manifest.json` records:

```text
model version
architecture version
weights hash
tokenizer hash
supported backends
quantization
training dataset manifest hash
```

---

# 37. Training artifact flow

```text
PyTorch training
        ↓
checkpoint
        ↓
SafeTensors export
        ↓
Rust parity test
        ↓
local hardware benchmarks
        ↓
release candidate
```

ONNX, TensorRT and CoreML files are derived artifacts.

---

# 38. Local-first acceptance criterion

Before cloud deployment, the same released model must run on:

```text
M4 Max
M5
Windows NVIDIA
```

using the same:

```text
model.safetensors
config.json
tokenizer.json
```

No retraining per platform.

Quantized variants may differ.

---

# 39. Cloud acceptance criterion

The same Rust inference service must:

```text
run locally
run in Docker
run on RunPod
```

without changing model semantics.

Environment differences should be configuration only.

---

# 40. Custom kernels

Only after profiling.

Likely future candidates:

```text
fused option scoring
small-set attention
state-memory cross attention
quantized linear layers
batched pointer head
```

Implement CPU reference first, then optimized CUDA/Metal versions behind the same abstraction.

---

# 41. Final recommendation

Start with exactly this stack:

```text
MODEL RESEARCH
Python + PyTorch

WEIGHTS
SafeTensors

TOKENIZATION
HF Tokenizers

LOCAL INFERENCE
Rust + Candle
Metal on M4/M5
CUDA on RTX
CPU fallback

LOCAL/API SERVER
Rust + Axum + Tokio

CUDA OPTIMIZATION
ONNX Runtime / TensorRT when benchmarks justify it

APPLE PERFORMANCE REFERENCE
MLX / CoreML

CLOUD INFERENCE
Docker + RunPod Serverless
scale-to-zero

CLOUD TRAINING
RunPod Pods only when local hardware is insufficient

FUTURE GLOBAL SERVICE
same Rust/Docker runtime
multiple regions
global traffic router
```

The rule is:

> **Rust owns the product. Python owns experimentation until Rust training is equally productive. SafeTensors is the contract between both worlds.**

This keeps research fast while ensuring that JevClone ultimately runs as a native, compact, high-performance decision engine on M4/M5, Windows CUDA and cloud GPU infrastructure without requiring Python in production.
