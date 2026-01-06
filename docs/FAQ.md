# GRAIL Miner FAQ

Frequently asked questions for GRAIL miners on Bittensor subnet 81.

---

## 1. What GPU should I use?

**Minimum requirements** (compute.min.yaml:1):
- **GPU**: NVIDIA A100 40GB/80GB
- **VRAM**: 24GB minimum, 40GB recommended
- **Compute Capability**: 9.0+ (Hopper architecture)
- **CPU**: 16+ cores @ 3.2GHz
- **RAM**: 64GB minimum
- **Storage**: 100GB SSD (50K IOPS)

**For competitive mining**: Multiple GPUs (2-8+ A100s) required to generate enough rollouts.

See compute.min.yaml for complete specifications.

---

## 2. How do I debug and optimize my miner?

**Don't debug on mainnet!** Run a local validator instead.

**Quick steps:**
1. Run local validator: `grail -vv validate > validate.log 2>&1 &`
2. Run your miner: `grail -vv mine > mine.log 2>&1 &`
3. Monitor: `tail -f validate.log | grep "uid=<your-uid>"`
4. Reduce ban time: Set `FAILURE_LOOKBACK_WINDOWS = 1` in grail/shared/constants.py:223

**Full guide**: [miner-debugging.md](miner-debugging.md)

**Optimization libraries**: vLLM, TensorRT-LLM, DeepSpeed, Flash Attention

See [vllm-setup.md](vllm-setup.md) for integration.

---

## 3. Is running the base code enough to earn incentives?

**No.** Base code is 10-20x too slow.

**Requirements** (grail/shared/constants.py:187, grail/shared/constants.py:204):
- Target: ~5,120 unique rollouts per window (6 minutes)
- Rate needed: ~14 rollouts/second sustained
- Base code: ~1-2 rollouts/second (insufficient)

**To compete, you need:**
- Multiple GPUs (2-8+)
- Optimized inference (vLLM, SGLang)
- Parallelization and batching
- Profiling and tuning

**Base code is a starting point, not production.**

---

## 4. How are incentives structured?

**Based on unique rollouts** over 12 windows (grail/scoring/weights.py:34):

**Scoring formula:**
1. Count unique rollouts (cap: 61,440 total across 12 windows)
2. Apply superlinear: `score = unique^4.0` (grail/shared/constants.py:179)
3. Normalize against cap
4. Apply 80% burn to UID 0 (grail/shared/constants.py:197)
5. Distribute remaining 20% to miners

**Key insight**: Superlinear exponent (4.0) means **2x rollouts = 16x score**. Volume dominates.

**Failure penalties** (grail/scoring/weights.py:75):
- Any failure in last 14 windows → zero weight
- Zero activity → zero weight
- Never validated → zero weight

**Example scores:**
| Rollouts (12 windows) | Relative Score |
|-----------------------|----------------|
| 5,000                 | 1x             |
| 10,000                | 16x            |
| 30,000                | 1,296x         |

Top miners near the 61,440 cap dominate weight distribution.

---

## 5. How many rollouts are top miners generating?

**Check the public dashboards:**

**WandB** (README.md:27): https://wandb.ai/tplr/grail
- Filter by validator runs
- Look for metrics: `miner/{uid}/rollouts_per_window`, `miner/{uid}/unique_rollouts`

**Grafana** (README.md:30): https://grail-grafana.tplr.ai/
- Real-time logs and network stats

**Typical top miner performance:**
- 3,000-8,000 rollouts per window
- 30,000-60,000+ unique rollouts per 12 windows
- 90%+ validation success rate

**To compete**: Match or exceed top miner throughput using multi-GPU setups and optimized inference.

---

## Additional Resources

- **Miner Setup**: [miner.md](miner.md)
- **Validator Setup**: [validator.md](validator.md)
- **Debugging**: [miner-debugging.md](miner-debugging.md)
- **vLLM Integration**: [vllm-setup.md](vllm-setup.md)
- **GitHub**: https://github.com/one-covenant/grail/issues
- **Discord**: https://discord.com/channels/799672011265015819/1354089114189955102
