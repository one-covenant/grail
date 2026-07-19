"""Preflight + smoke for grail delta checkpoint bench (goal.md §4 + §5.1).

Validates, in order:
1. grail.infrastructure.delta_checkpoint imports (4 functions)
2. xxhash / safetensors / torch versions
3. Tiny synthetic round-trip (doc §4)
4. Smoke: 8-layer 512-hidden synthetic model, perturb 10%, full
   encode -> safetensors write (CPFS) -> read -> apply -> bit-exact hash verify
5. HF Qwen3-30B-A3B config load
6. CPFS free space > 100 GB

Exit 0 only if all pass.
"""

import json
import os
import shutil
import sys
import time

sys.path.insert(0, "/storage/openpsi/users/pengzai.pyq/grail")

print("=== [1/6] import grail delta module ===")
from grail.infrastructure.delta_checkpoint import (  # noqa: E402
    apply_sparse_delta,
    compute_sparse_delta,
    compute_weights_hash,
    estimate_sparse_size,
)

print("OK: 4 functions imported")

print("\n=== [2/6] xxhash / safetensors / torch sanity ===")
import torch  # noqa: E402
import xxhash  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import safetensors  # noqa: E402

print("xxhash:", xxhash.VERSION)
print("safetensors:", safetensors.__version__)
print("torch:", torch.__version__)

print("\n=== [3/6] tiny synthetic round-trip ===")
g = torch.Generator().manual_seed(1234)
base = {"w1": torch.randn(100, 200, generator=g), "w2": torch.randn(50, 50, generator=g)}
current = {k: v.clone() for k, v in base.items()}
current["w1"].view(-1)[torch.randperm(100 * 200, generator=g)[:100]] = 0.5
sparse, shapes, stats = compute_sparse_delta(current, base)
print(f"stats: total={stats['total_params']} changed={stats['nonzero_params']}")
assert stats["nonzero_params"] == 100, f"expected 100 changed, got {stats['nonzero_params']}"
reconstructed = apply_sparse_delta(base, sparse, shapes)
ok = all(torch.equal(current[k].cpu(), reconstructed[k].cpu()) for k in current)
print("tiny reconstruct OK:", ok)
assert ok

print("\n=== [4/6] smoke: 8-layer 512-hidden bf16 model, 10% perturb, disk round-trip ===")
smoke_dir = os.environ.get(
    "GRAIL_SMOKE_DIR", "/storage/openpsi/users/pengzai.pyq/grail_bench/smoke"
)
os.makedirs(smoke_dir, exist_ok=True)

g2 = torch.Generator().manual_seed(42)
smoke_base = {}
for layer in range(8):
    for pname, shape in [
        ("attn.qkv", (512, 1536)),
        ("attn.o", (512, 512)),
        ("mlp.up", (512, 2048)),
        ("mlp.down", (2048, 512)),
    ]:
        smoke_base[f"layers.{layer}.{pname}"] = torch.randn(
            *shape, generator=g2, dtype=torch.float32
        ).to(torch.bfloat16)
smoke_base["embed"] = torch.randn(32000, 512, generator=g2).to(torch.bfloat16)
total_elems = sum(t.numel() for t in smoke_base.values())
print(f"smoke model: {len(smoke_base)} tensors, {total_elems / 1e6:.1f}M params")

# Perturb 10% of elements via lowest-mantissa-bit flip (guaranteed bf16 change)
smoke_cur = {}
for k, t in smoke_base.items():
    c = t.clone()
    mask = torch.rand(c.shape, generator=g2) < 0.10
    c.view(torch.int16)[mask] ^= 1
    smoke_cur[k] = c

t0 = time.perf_counter()
sp, sh, st = compute_sparse_delta(smoke_cur, smoke_base)
t_encode = time.perf_counter() - t0
density = st["nonzero_params"] / st["total_params"]
print(f"encode: {t_encode:.3f}s, density={density:.4f} (expect ~0.10)")
assert 0.08 < density < 0.12, f"density {density} outside [0.08, 0.12]"

delta_path = os.path.join(smoke_dir, "smoke_delta.safetensors")
shapes_path = os.path.join(smoke_dir, "smoke_shapes.json")
t0 = time.perf_counter()
save_file(sp, delta_path)
with open(shapes_path, "w") as f:
    json.dump(sh, f)
t_write = time.perf_counter() - t0
size_mb = os.path.getsize(delta_path) / 2**20
print(f"write: {t_write:.3f}s, delta file {size_mb:.1f} MiB")

t0 = time.perf_counter()
sp_loaded = load_file(delta_path)
with open(shapes_path) as f:
    sh_loaded = json.load(f)
t_read = time.perf_counter() - t0

t0 = time.perf_counter()
recon = apply_sparse_delta(smoke_base, sp_loaded, sh_loaded)
t_apply = time.perf_counter() - t0
print(f"read: {t_read:.3f}s, apply: {t_apply:.3f}s")

h_cur = compute_weights_hash(smoke_cur)
h_rec = compute_weights_hash(recon)
print(f"hash current={h_cur}\nhash recon  ={h_rec}")
assert h_cur == h_rec, "SMOKE FAIL: hash mismatch after disk round-trip"
print("smoke bit-exact: PASS")
est = estimate_sparse_size(st["nonzero_params"], value_dtype=torch.bfloat16)
print(f"estimate_sparse_size: {est / 2**20:.1f} MiB (actual file {size_mb:.1f} MiB)")

for p in (delta_path, shapes_path):
    os.remove(p)

print("\n=== [5/6] HF model config load ===")
from transformers import AutoConfig  # noqa: E402

c = AutoConfig.from_pretrained(
    "/storage/openpsi/models/Qwen__Qwen3-30B-A3B", trust_remote_code=True
)
print("model:", c.__class__.__name__, "layers:", c.num_hidden_layers)

print("\n=== [6/6] CPFS free ===")
_, _, free = shutil.disk_usage("/storage")
print(f"CPFS free: {free // 2**30} GiB")
assert free > 100 * 2**30, "CPFS free < 100 GiB"

print("\nALL PREFLIGHT + SMOKE CHECKS PASSED")
