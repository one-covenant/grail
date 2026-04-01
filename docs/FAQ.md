# GRAIL Miner FAQ

Frequently asked questions for GRAIL miners on Bittensor subnet 81.

---

## 1. Is the subnet currently active for mining?

**Yes.** The subnet is live and accepting miners, but there are few active miners right now because it is extremely competitive.

**Key facts:**
- This is not an easy subnet. You need to be deeply technical and able to change the code fundamentally.
- Becoming competitive is not a 1-day optimization. It requires multi-node coordination, compute/comms overlap, and many levels of efficiency work.
- If you reach the rollout targets and mine faithfully, you will earn rewards.

**Before starting**, carefully read:
- [bounties.md](bounties.md) for open optimization bounties
- [miner-debugging.md](miner-debugging.md) for debugging and local testing

---

## 2. What GPU should I use?

**Minimum requirements** (compute.min.yaml:1):
- **GPU**: NVIDIA A100 40GB/80GB
- **VRAM**: 24GB minimum, 40GB recommended
- **Compute Capability**: 9.0+ (Hopper architecture)
- **CPU**: 16+ cores @ 3.2GHz
- **RAM**: 64GB minimum
- **Storage**: 100GB SSD (50K IOPS)

**GPU roles:**
- **Proof verification**: Requires A100s (compute capability 9.0+).
- **Inference/decoding**: Can use other GPUs.

**For competitive mining**: Multiple GPUs (2-8+ A100s) required to generate enough rollouts, with multi-node setups for top performance.

See compute.min.yaml for complete specifications.

---

## 3. How do I debug and optimize my miner?

**Don't debug on mainnet!** Run a local validator instead.

**Quick steps:**
1. Run local validator: `grail -vv validate > validate.log 2>&1 &`
2. Run your miner: `grail -vv mine > mine.log 2>&1 &`
3. Monitor: `tail -f validate.log | grep "uid=<your-uid>"`
4. Reduce ban time: Set `FAILURE_LOOKBACK_WINDOWS = 1` in grail/protocol/constants.py:223

**Full guide**: [miner-debugging.md](miner-debugging.md)

**Optimization libraries**: vLLM, TensorRT-LLM, DeepSpeed, Flash Attention

See [vllm-setup.md](vllm-setup.md) for integration.

---

## 4. Is running the base code enough to earn incentives?

**No.** Base code generates ~16 rollouts per window. You need ~1,500 unique rollouts per window to hit the cap.

**Requirements** (grail/protocol/constants.py:136):
- Cap: 1,500 unique rollouts per window (18,000 across 12 windows)
- Rate needed: ~4 rollouts/second sustained over a 6-minute window
- Base code: ~16 rollouts/window (~0.04/second, insufficient)

**To compete, you need:**
- Multiple GPUs (2-8+), potentially multi-node setups
- Optimized inference (vLLM, SGLang)
- Multi-node coordination with compute/comms overlap
- Parallelization, batching, profiling, and tuning

**Base code is a starting point, not production.**

---

## 5. How are incentives structured?

**Based on unique rollouts** over 12 windows (grail/scoring/weights.py:34):

**Scoring formula (capped mode, current default):**
1. Count unique rollouts per window, capped at 1,500 (grail/protocol/constants.py:136)
2. Period cap: 1,500 x 12 = 18,000 across the rolling interval
3. Apply superlinear: `score = unique^4.0` (grail/protocol/constants.py:131)
4. Normalize against cap
5. Apply 80% burn to UID 0 (grail/protocol/constants.py:142)
6. Distribute remaining 20% to miners

**Key insight**: Producing more than 1,500 unique rollouts per window gives no additional weight. Below the cap, the superlinear exponent (4.0) means **2x rollouts = 16x score**.

**Failure penalties** (grail/scoring/weights.py:75):
- Any failure in last 14 windows → zero weight
- Zero activity → zero weight
- Never validated → zero weight

See [incentive-mechanism.md](incentive-mechanism.md) for the full verification pipeline and scoring details.

---

## 6. How many rollouts are top miners generating?

**Check the public dashboards:**

**WandB** (README.md:27): https://wandb.ai/tplr/grail
- Filter by validator runs
- Look for metrics: `miner/{uid}/rollouts_per_window`, `miner/{uid}/unique_rollouts`

**Grafana** (README.md:30): https://grail-grafana.tplr.ai/
- Real-time logs and network stats

**Typical top miner performance:**
- 1,000-1,500 unique rollouts per window (cap is 1,500)
- 12,000-18,000 unique rollouts per 12 windows (period cap is 18,000)
- 90%+ validation success rate

**To compete**: Consistently hit the 1,500/window cap using multi-GPU setups and optimized inference.

---

## Additional Resources

- **Miner Setup**: [miner.md](miner.md)
- **Validator Setup**: [validator.md](validator.md)
- **Debugging**: [miner-debugging.md](miner-debugging.md)
- **vLLM Integration**: [vllm-setup.md](vllm-setup.md)
- **GitHub**: https://github.com/one-covenant/grail/issues
- **Discord**: https://discord.com/channels/799672011265015819/1354089114189955102
