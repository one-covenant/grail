### Delta Weight Communication Protocol (DWCP)

This document specifies a reusable **delta weight communication protocol** for settings where only a small fraction of model parameters change “meaningfully” per update (e.g., **RLVR/GRPO**-style policy updates, adapter-only training, quantized-weight updates). The protocol is designed to work in two modes:

- **Broadcast / replication**: trainer → inference workers (or miners) for fast model refresh.
- **Collective / synchronization**: many trainers exchange sparse **updates** (weights or gradients) in a DDP-like fashion.

It is transport-agnostic: the same payload can be carried via object storage (pull), gRPC (push/stream), or any message bus.

---

### 1) Goals and non-goals

- **Goals**
  - **30×+ bandwidth reduction** relative to dense FP16 checkpoints when “effective changes” are sparse (≈1–5%).
  - **Correctness and recoverability**: receivers can verify they reconstructed exactly the intended target state or detect mismatch and resync.
  - **Reusability** across:
    - process boundaries (trainer ↔ inference server),
    - machine boundaries (cluster),
    - many-to-many settings (multi-trainer).
  - **Incremental deployment**: deltas are a fast-path; dense checkpoints remain the fallback/anchor.

- **Non-goals**
  - Not a replacement for general distributed training collectives in the worst case (dense updates).
  - Not tied to a specific optimizer; it can carry either **weights** or **gradients**.

---

### 2) Core idea: “patch” semantics, not “add” semantics

There are two delta semantics:

- **A. Patch (absolute values at selected indices)** *(recommended for replication)*  
  For each parameter tensor, transmit a set of indices and the **new values** at those indices. Receivers reconstruct by **direct assignment** at those positions (no floating-point accumulation).
  - **Pros**: bit-exact reconstruction (given identical base), no numeric drift, easy hash verification.
  - **Cons**: harder to combine updates from multiple senders (needs conflict resolution / averaging policy).

- **B. Add (true deltas / gradients)** *(recommended for collective training)*  
  Transmit sparse deltas or gradients; receivers apply via addition/optimizer step.
  - **Pros**: natural for all-reduce / averaging; supports error-feedback/residual accumulation.
  - **Cons**: numeric drift, needs careful handling with mixed precision.

In practice, implement **both** codecs under one interface and choose per use case.

---

### 3) Versioned state model (required for correctness)

Every update is defined relative to a base model version.

- **ModelVersion**
  - `model_id`: identifies architecture + parameter naming.
  - `version_id`: monotonic step/window/iteration.
  - `base_hash`: hash of the *base* state (or anchor) the update expects.
  - `target_hash`: hash of the intended post-apply state (for patch mode) or an optional integrity checksum.

**Receiver rule:** if `base_hash` does not match local state, the receiver must **resync** (fetch an anchor FULL checkpoint or request a chain rebuild).

**Practical deployment:** keep periodic **FULL anchors** (e.g., every K windows) and chain deltas in-between to bound recovery cost.

---

### 4) Payload encoding (DWCP codec)

The payload is a set of tensor updates. Recommended default is **COO sparse encoding per tensor**:

- For each parameter `W` flattened to 1D:
  - `W.indices`: int32/int64 indices of changed entries (sorted ascending).
  - `W.values`: the corresponding values (patch mode) or deltas/gradients (add mode).
  - `W.shape`: stored in metadata (to reshape on load).

**Compression layering (orthogonal choices):**

- **Index coding**
  - *Baseline*: int32 indices.
  - *Better*: delta-code indices (store differences), then varint + entropy coding.
  - *Best for GPUs*: **block-sparse** indices (block_id + offset) to enable kernel-friendly scatter.

- **Value coding**
  - *Lossless*: fp16/bf16 (or original dtype) with general-purpose compression (zstd).
  - *Lossy*: int8/fp8 quantization with per-tensor or per-block scale; optional stochastic rounding.

- **Container**
  - Practical: `safetensors` for portability and easy key/value packing.
  - Streaming: chunked framing (see §6) for long tensors and backpressure.

---

### 5) What counts as a “changed weight” (the sparsity lever)

In standard SGD, *almost every parameter changes* at float-level. To achieve 1–5% “effective changes”, you need a **selection rule** that makes sparsity meaningful:

- **Trainable-subset rule**
  - Update only adapters (LoRA), only specific layers, only heads, etc.
  - Extremely high leverage; avoids scanning all parameters.

- **Threshold rule**: include positions where \(|\Delta| > \tau\)
  - Simple, but tuning \(\tau\) trades bandwidth vs quality.

- **Top-k rule**: include the k largest-magnitude changes per tensor (or globally)
  - Stable update size; but requires selection overhead and coordination in multi-trainer mode.

- **Quantized-space change rule** *(often best for RLVR/GRPO sampling loops)*  
  Quantize weights to a communication dtype/grid; mark “changed” only if the **quantized** value differs from base. This naturally produces sparsity when changes are small.

---

### 6) Protocol messages (transport-agnostic)

DWCP defines logical messages; transports can map them to files (object store), RPCs, or streams.

- **HELLO**
  - Sent by receiver: `{model_id, have_version_id, have_hash, supported_codecs, max_chunk_bytes}`

- **OFFER**
  - Sent by sender: `{model_id, base_version_id, base_hash, target_version_id, target_hash?, codec, estimated_bytes, chunking_plan}`

- **REQUEST**
  - Receiver requests either: `DELTA` from a base it has, or `ANCHOR` (full) if mismatch.

- **DELTA_CHUNK**
  - `{update_id, chunk_id, tensor_name, indices_bytes, values_bytes, codec_params, crc32}`

- **COMMIT**
  - `{update_id, target_hash, stats, signature?}`  
  Receiver applies only if all chunks verify and (optionally) signature is valid.

- **NACK / RESYNC**
  - Receiver asks for anchor (full) or earlier delta chain.

**Chunking guidance:** chunk by tensor blocks (e.g., 1–16MB compressed) so receivers can apply progressively and avoid large peak memory.

---

### 7) Mode A: trainer → inference (replication)

**Default recommendation: patch mode (absolute values).**

#### A0) Mapping to the current GRAIL implementation (what already exists)

GRAIL already implements a concrete instance of DWCP in its checkpoint pipeline:

- **Codec (patch/COO)**: `grail/infrastructure/delta_checkpoint.py` stores `{param}.indices` + `{param}.values` and applies by direct assignment (no arithmetic).
- **Producer**: `grail/trainer/checkpoint_publisher.py` can publish `CHECKPOINT_TYPE_DELTA` updates (compressed with zstd) and periodically `CHECKPOINT_TYPE_FULL` anchors.
- **Consumer**: `grail/infrastructure/checkpoint_consumer.py` can:
  - apply a single delta in-place (`apply_delta_in_place`) when `prev_window` matches,
  - reconstruct a chain from an anchor when recovering from missed windows,
  - verify `weights_hash` to detect mismatch/corruption.

This document generalizes that design to additional transports (e.g., streaming) and to multi-trainer synchronization.

#### A1) Pull-based (object store) replication

- Trainer publishes:
  - FULL anchor checkpoint occasionally.
  - DELTA updates each window/iteration (small).
  - READY marker or committed manifest (atomicity).
- Inference workers:
  - Poll for latest ready update.
  - If current version matches `prev_window/base_hash`, apply delta in-place.
  - Else fetch anchor FULL and replay chain.

**Tradeoff**: higher latency (polling) but simplest operationally; great for large fanout.

#### A2) Push/stream replication

- Trainer streams `DELTA_CHUNK`s to inference servers over gRPC.
- Inference servers apply in background and “swap” to new version when COMMIT arrives.

**Tradeoff**: lower latency but needs connection management, backpressure, and retry semantics.

#### Correctness knobs for RLVR/GRPO

- **Staleness tolerance**: inference can sample with slightly stale weights; attach `version_id` to rollouts so trainer can account for policy lag (optional).
- **Anchors**: periodic full anchors bound worst-case recovery for newly started inference servers.

---

### 8) Mode B: multi-trainer (DDP-like) synchronization

Multi-trainer is harder because updates must be **combined**. There are three practical approaches, each with different tradeoffs.

#### B1) Sparse all-reduce on a shared mask (structured sparsity)

- All ranks use the *same* index set each step (e.g., random but deterministic mask per step; or block-sparse pattern per layer).
- Each rank communicates only values for those indices.
- Combine via standard all-reduce (sum/avg).

**Pros**: fast, simple collectives, stable comm volume.  
**Cons**: mask may not match true top-k; may waste bandwidth on “unchanged” entries; may reduce convergence if too aggressive.

#### B2) Top-k per rank + union merge (unstructured sparsity)

Pipeline:
- Each rank picks local top-k indices (by |grad| or |delta|).
- Communicate indices (allgather); compute union; communicate values for union; aggregate.

**Pros**: focuses bandwidth on salient updates.  
**Cons**: union can explode (density rises with #ranks); needs multiple rounds; complex to implement efficiently.

#### B3) Parameter-server / aggregator (central or hierarchical)

- Each trainer sends sparse updates to an aggregator.
- Aggregator merges (sum/avg), optionally re-sparsifies, then broadcasts back.

**Pros**: easy to enforce global top-k; supports asymmetric links; good for heterogeneous clusters.  
**Cons**: aggregator bottleneck; needs careful fault tolerance; less “pure DDP”.

#### Required convergence stabilization for sparse training comms

For add-mode gradient sparsification, typical choices:

- **Error feedback / residual accumulation**: keep unsent gradient mass and add it next step to reduce bias.
- **Momentum correction**: synchronize momentum buffers occasionally or use local momentum with periodic full sync.

---

### 9) Paper-facing tradeoffs (what to emphasize)

- **Patch vs add semantics**
  - Patch: best for inference replication (deterministic, verifiable).
  - Add: best for training collectives (mergeable).

- **Index selection**
  - Threshold: simple, but comm volume varies and can be unstable across layers.
  - Top-k: fixed budget, but expensive/complex and problematic at scale (#ranks union).
  - Structured masks: fastest, but may reduce quality.
  - Quantized-space changes: often a sweet spot (sparsity without heavy selection).

- **Encoding**
  - COO: easiest, but not GPU-kernel friendly.
  - Block-sparse: faster apply, better compression of indices, but less flexible sparsity shape.

- **Verification**
  - Full-state hash: strong correctness (detects wrong base, corruption, drift).
  - Per-chunk CRC: fast transport integrity; combine with signature if adversarial environment.

- **Recovery**
  - Longer delta chains reduce bandwidth but increase cold-start time and risk; anchors bound this.

---

### 10) Bandwidth math (why 30× is plausible)

Let:
- \(N\) = total parameters
- dense checkpoint cost ≈ \(2N\) bytes (FP16 weights only)
- sparse patch cost ≈ \(pN \cdot (\text{index_bytes} + \text{value_bytes})\)

With int32 indices (4B) and FP16 values (2B): sparse bytes ≈ \(pN \cdot 6\).

Relative to dense FP16: \(\frac{pN \cdot 6}{2N} = 3p\).

Examples:
- \(p=1\%\): \(3p=0.03\) → **33× reduction** (before compression).
- \(p=2\%\): \(3p=0.06\) → **16.7× reduction** (before compression).
- zstd + index delta-coding often gives an additional ~1.5–3× on top, making **30×** realistic in the \(p \approx 1–2\%\) regime.

---

### 11) Recommended reusable module design

Design the implementation as 3 orthogonal layers:

- **State/version layer**
  - Computes/validates `base_hash`/`target_hash`, manages anchors, chain length, and resync policy.

- **Codec layer**
  - Pluggable: `PatchCOO`, `DeltaCOO`, `BlockSparse`, `Quant8`, etc.
  - Exposes: `encode(base, target) -> chunks`, `decode_apply(base, chunks) -> target`.

- **Transport layer**
  - Pluggable: `ObjectStoreTransport`, `GrpcTransport`, `NCCLCollectiveTransport`.
  - Exposes: `offer/request/stream/commit` semantics and retry/backpressure.

This separation keeps the “protocol” reusable across projects and deployments, while allowing each environment to choose an appropriate transport.


