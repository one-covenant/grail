#!/usr/bin/env python3
"""
Benchmark vLLM vs HF generation for GRAIL miner defaults.

- No flags required; uses env defaults from grail/shared/constants.py
- Renders SAT prompt via the miner's chat template
- Uses env-based backend selection (INFERENCE_BACKEND, VLLM_*)
- Measures GRPO-style multi-rollouts (n=GRAIL_ROLLOUTS_PER_PROBLEM)
- Also runs a single HF forward pass to estimate proof overhead

Run:
  uv run python scripts/benchmark_vllm.py
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, List

import torch

from grail.grail import dot_mod_q, r_vec_from_randomness  # proof math
from grail.shared.constants import (
    MODEL_NAME,
    MAX_NEW_TOKENS,
    ROLLOUTS_PER_PROBLEM,
    VLLM_BASE_URL,
    VLLM_MODEL,
    VLLM_TIMEOUT_S,
    VLLM_MAX_RETRIES,
)
from grail.mining.rollout_generator import _build_qwen_chat_template, SYSTEM_PROMPT, REASONING_START


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("benchmark")


def build_prompt(tokenizer: Any, problem_text: str) -> str:
    try:
        tpl = _build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)
        if getattr(tokenizer, "chat_template", None) != tpl:
            tokenizer.chat_template = tpl
    except Exception:
        pass
    messages = [{"role": "user", "content": problem_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def fake_sat_problem(seed: str = "bench-seed", num_vars: int = 6, num_clauses: int = 12) -> str:
    # Minimal SAT text to exercise the miner prompt path (does not validate)
    lines = [f"SAT Problem (seed: {seed[:8]}...):", f"Variables: {num_vars}", "Clauses:"]
    for i in range(num_clauses):
        lines.append(f"  (x1 OR NOT x2 OR x3)")
    instr = (
        "Provide your final assignment between <SOLUTION></SOLUTION> as "
        "space-separated 0/1 values for x1..xN (e.g., <SOLUTION>0 1 0 1</SOLUTION>).\n"
    )
    return "\n".join(["SAT Problem:", "\n".join(lines), instr])


def run_vllm(prompt: str, n: int) -> List[str]:
    from grail.inference.vllm_client import VLLMClient

    client = VLLMClient(
        base_url=VLLM_BASE_URL,
        model=(VLLM_MODEL or None),
        timeout=float(VLLM_TIMEOUT_S),
        max_retries=int(VLLM_MAX_RETRIES),
    )
    comps = client.generate(
        prompt,
        n=n,
        max_tokens=int(MAX_NEW_TOKENS),
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        stop=None,
        ignore_eos=False,
    )
    return [c.text for c in comps]


def run_hf(model: Any, tokenizer: Any, prompt: str, n: int) -> List[str]:
    # Sequential HF generations to mimic miner fallback
    outputs: List[str] = []
    tok = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    input_ids = tok.input_ids.to(model.device)
    attn = tok.attention_mask.to(model.device)
    for _ in range(n):
        with torch.inference_mode():
            outs = model.generate(
                input_ids,
                attention_mask=attn,
                max_new_tokens=int(MAX_NEW_TOKENS),
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        seq = outs.sequences[0].tolist()
        prompt_len = int(input_ids.shape[1])
        completion_ids = seq[prompt_len:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        outputs.append(text)
    return outputs


def maybe_proof_pass(model: Any, tokenizer: Any, prompt: str, completions: List[str]) -> float:
    # Estimate time/VRAM for one proof forward pass using the longest completion
    tok = tokenizer(prompt, return_tensors="pt")
    prefix = tok.input_ids[0].tolist()
    comp = max(completions, key=lambda t: len(t)) if completions else ""
    comp_ids = tokenizer(comp, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    all_ids = prefix + comp_ids
    randomness_hex = "deadbeef" * 8
    r_vec = r_vec_from_randomness(randomness_hex, getattr(model.config, "hidden_size", 4096))
    start = time.time()
    with torch.inference_mode():
        token_tensor = torch.tensor([all_ids], dtype=torch.long, device=model.device)
        outs = model(token_tensor, output_hidden_states=True)
        h_layer = outs.hidden_states[-1][0]
        _svals = [dot_mod_q(h_layer[pos], r_vec) for pos in range(min(len(all_ids), h_layer.size(0)))]
    elapsed = time.time() - start
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vLLM vs HF miner defaults")
    # Keep defaults fixed to simplify usage per request
    args = parser.parse_args([])

    # Load base model/tokenizer via Prover (no wallet needed here, but Prover enforces one)
    # Workaround: create a dummy-like wallet by bypassing Prover; load directly instead.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True)
        .to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        .eval()
    )

    n = int(ROLLOUTS_PER_PROBLEM)
    problems = 2  # fixed per request
    logger.info(f"Benchmarking both backends | model={MODEL_NAME} n={n} max_new_tokens={MAX_NEW_TOKENS}")

    results = {}

    # vLLM backend
    if VLLM_BASE_URL:
        v_gen_times = []
        v_proof_times = []
        v_total_comps = 0
        v_start = time.time()
        for i in range(problems):
            sat_text = fake_sat_problem(seed=f"bench-{i}")
            prompt = build_prompt(tokenizer, sat_text)
            t0 = time.time()
            try:
                outs = run_vllm(prompt, n)
            except Exception as e:
                logger.warning(f"vLLM failed on group {i+1}: {e}")
                outs = []
            dt = time.time() - t0
            v_gen_times.append(dt)
            v_total_comps += len(outs)
            proof_dt = maybe_proof_pass(model, tokenizer, prompt, outs)
            v_proof_times.append(proof_dt)
        v_total_time = time.time() - v_start
        v_rps_gen = v_total_comps / sum(v_gen_times) if sum(v_gen_times) > 0 else 0.0
        v_rps_e2e = v_total_comps / (sum(v_gen_times) + sum(v_proof_times)) if (sum(v_gen_times) + sum(v_proof_times)) > 0 else 0.0
        results["vllm"] = {
            "gen_times": v_gen_times,
            "proof_times": v_proof_times,
            "total_comps": v_total_comps,
            "total_time": v_total_time,
            "rps_gen": v_rps_gen,
            "rps_e2e": v_rps_e2e,
        }
    else:
        logger.info("VLLM_BASE_URL not set; skipping vLLM backend")

    # HF backend
    h_gen_times = []
    h_proof_times = []
    h_total_comps = 0
    h_start = time.time()
    for i in range(problems):
        sat_text = fake_sat_problem(seed=f"bench-{i}")
        prompt = build_prompt(tokenizer, sat_text)
        t0 = time.time()
        outs = run_hf(model, tokenizer, prompt, n)
        dt = time.time() - t0
        h_gen_times.append(dt)
        h_total_comps += len(outs)
        proof_dt = maybe_proof_pass(model, tokenizer, prompt, outs)
        h_proof_times.append(proof_dt)
    h_total_time = time.time() - h_start
    h_rps_gen = h_total_comps / sum(h_gen_times) if sum(h_gen_times) > 0 else 0.0
    h_rps_e2e = h_total_comps / (sum(h_gen_times) + sum(h_proof_times)) if (sum(h_gen_times) + sum(h_proof_times)) > 0 else 0.0
    results["hf"] = {
        "gen_times": h_gen_times,
        "proof_times": h_proof_times,
        "total_comps": h_total_comps,
        "total_time": h_total_time,
        "rps_gen": h_rps_gen,
        "rps_e2e": h_rps_e2e,
    }

    # Print comparison
    logger.info("\n=== Detailed per-group timings ===")
    for i in range(problems):
        vg = results.get("vllm", {}).get("gen_times", [None] * problems)[i] if "vllm" in results else None
        vp = results.get("vllm", {}).get("proof_times", [None] * problems)[i] if "vllm" in results else None
        hg = results["hf"]["gen_times"][i]
        hp = results["hf"]["proof_times"][i]
        if vg is not None:
            logger.info(
                f"group {i+1}: vLLM gen={vg:.3f}s proof={vp:.3f}s e2e={(vg + vp):.3f}s | HF gen={hg:.3f}s proof={hp:.3f}s e2e={(hg + hp):.3f}s"
            )
        else:
            logger.info(
                f"group {i+1}: HF gen={hg:.3f}s proof={hp:.3f}s e2e={(hg + hp):.3f}s (vLLM skipped)"
            )

    logger.info("\n=== Summary ===")
    if "vllm" in results:
        r = results["vllm"]
        logger.info(
            f"vLLM: {r['total_comps']} comps | gen_sum={sum(r['gen_times']):.3f}s proof_sum={sum(r['proof_times']):.3f}s "
            f"e2e_sum={(sum(r['gen_times']) + sum(r['proof_times'])):.3f}s | gen_rps={r['rps_gen']:.2f} e2e_rps={r['rps_e2e']:.2f}"
        )
    r = results["hf"]
    logger.info(
        f"HF  : {r['total_comps']} comps | gen_sum={sum(r['gen_times']):.3f}s proof_sum={sum(r['proof_times']):.3f}s "
        f"e2e_sum={(sum(r['gen_times']) + sum(r['proof_times'])):.3f}s | gen_rps={r['rps_gen']:.2f} e2e_rps={r['rps_e2e']:.2f}"
    )


if __name__ == "__main__":
    main()


