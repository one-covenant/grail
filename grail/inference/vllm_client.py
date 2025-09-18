#!/usr/bin/env python3
"""vLLM HTTP client for high-throughput generation.

Supports OpenAI-compatible /v1/completions endpoint exposed by vllm serve.
We send a rendered prompt (already chat-templated) and request n completions
with token-level logprobs when available.

This client is intentionally lightweight and does not depend on the vllm
python package, only on 'requests'.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class VLLMCompletion:
    text: str
    token_logprobs: Optional[List[float]]


class VLLMClient:
    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("VLLM_API_KEY") or ""
        self.timeout = timeout
        self.max_retries = max_retries

        # Prefer OpenAI-compatible /v1/completions for simple prompt-based gen
        self.completions_url = f"{self.base_url}/v1/completions"

        self._session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(
        self,
        prompt: str,
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        ignore_eos: bool = True,
    ) -> List[VLLMCompletion]:
        """Generate n completions for a single prompt.

        Args:
            prompt: Fully rendered prompt string (after chat templating)
            n: Number of completions to return (GRPO group size)
            max_tokens: Maximum new tokens to sample
            temperature: Sampling temperature
            top_p: nucleus sampling
            repetition_penalty: Not universally supported in OpenAI schema; ignored if None
            stop: Optional stop strings

        Returns:
            List of VLLMCompletion with text and optional token-level logprobs
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "n": int(n),
            "logprobs": 1,  # request per-token logprobs when supported
            "echo": False,
        }
        # Encourage length-based termination to match miner's MAX_NEW_TOKENS path
        # when OpenAI-compatible wrapper supports it.
        payload["ignore_eos"] = bool(ignore_eos)
        if self.model:
            payload["model"] = self.model
        if stop:
            payload["stop"] = stop
        # repetition_penalty is not part of OpenAI schema; some vLLM deployments
        # accept it as an extension. Include when provided.
        if repetition_penalty is not None:
            payload["repetition_penalty"] = float(repetition_penalty)

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(
                    self.completions_url,
                    data=json.dumps(payload),
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"vLLM server HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                data = resp.json()
                choices = data.get("choices", [])
                results: List[VLLMCompletion] = []
                for ch in choices:
                    text = ch.get("text", "")
                    # OpenAI-style logprobs payload
                    lp = ch.get("logprobs")
                    token_logprobs: Optional[List[float]] = None
                    if isinstance(lp, dict) and isinstance(lp.get("token_logprobs"), list):
                        # Only completion tokens are included (no prompt echo)
                        token_logprobs = [
                            float(x) if x is not None else 0.0 for x in lp["token_logprobs"]
                        ]
                    results.append(VLLMCompletion(text=text, token_logprobs=token_logprobs))
                return results
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.warning(f"vLLM request failed (attempt {attempt+1}): {e}")
        # Exhausted retries
        raise RuntimeError(f"vLLM generation failed: {last_error}")


