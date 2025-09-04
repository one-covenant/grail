# Token Sampling Shape Check: A Beginner's Guide

This guide explains the math and intuition behind the "token sampling shape check" used to catch mismatches between the model that generated tokens and the validator model that evaluates them.

## What we are checking

- At each step of text generation, a language model produces a score for every vocabulary token. These scores are called logits.
- Turning logits into probabilities gives a distribution over the next token.
- If tokens are sampled from the same model as the validator, the probabilities that the validator assigns to the chosen tokens tend to be high and form a unimodal, "exponential‑ish" distribution with a mode near 1.
- If tokens are sampled from a different model but the validator only "prefills" context with a larger model, the probability distribution of chosen tokens often becomes bimodal: a chunk near 1 (still plausible) and a chunk near 0 (surprisingly unlikely), with fewer in the middle.

Our check collects these chosen‑token probabilities and tests whether their distribution looks unimodal-high versus bimodal (near 0 and near 1).

---

## From logits to probabilities

- Let $z \in \mathbb{R}^V$ be the logits over a vocabulary of size $V$. The softmax turns logits into probabilities:
$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}} \quad \text{for } i=1,\dots,V.$$

- In an autoregressive model, the token at position $t$ is predicted using context up to position $t-1$. If the actual chosen token at time $t$ is $x_t$, then the validator's probability for that chosen token is:
$$p_{\text{val},t} = p(x_t \mid z_{t-1}) = \text{softmax}(z_{t-1})[x_t].$$

- We evaluate only over completion tokens (not the prompt). If `prompt_length` is the number of prompt tokens and `completion_length` is the number of generated tokens, we compute
$$t = \text{prompt\_length}, \dots, \text{prompt\_length} + \text{completion\_length} - 1,$$
and at each such $t$ we use logits from index $t-1$.

Collect these into a list:
$$\{p_1, p_2, \dots, p_n\}, \quad \text{where } n=\text{completion\_length}.$$

---

## How the distribution should look

- If tokens were sampled from the same model as the validator, many $p_t$ will be relatively large, and the histogram of $p_t$ values is typically unimodal and skewed towards 1.
- If tokens were sampled from a different model, we often see many values near 1 (still plausible) and many near 0 (validator finds them unlikely), which yields a bimodal distribution with few values in the middle.

---

## Simple, interpretable statistics

Let $\alpha$ and $\beta$ be thresholds (defaults $\alpha=0.10$, $\beta=0.90$). Define:

- Low fraction (near 0):
$$\text{low\_frac} = \frac{1}{n}\sum_{t=1}^{n} \mathbf{1}\{p_t \le \alpha\}.$$
- High fraction (near 1):
$$\text{high\_frac} = \frac{1}{n}\sum_{t=1}^{n} \mathbf{1}\{p_t \ge \beta\}.$$
- Middle fraction:
$$\text{mid\_frac} = 1 - \text{low\_frac} - \text{high\_frac}.$$

Intuition:
- Large `low_frac` → many tokens look unlikely to the validator.
- Large `high_frac` → many tokens look very likely.
- Small `mid_frac` → thin middle mass, consistent with bimodality.

---

## Skewness, kurtosis, and the bimodality coefficient

We also compute shape moments of the $p_t$ values:

- Mean:
$$\mu = \frac{1}{n}\sum_{t=1}^{n} p_t.$$
- Centered values: $d_t = p_t - \mu$.
- Variance:
$$\sigma^2 = \frac{1}{n}\sum_{t=1}^{n} d_t^2.$$
- Third moment:
$$m_3 = \frac{1}{n}\sum_{t=1}^{n} d_t^3.$$
- Fourth moment:
$$m_4 = \frac{1}{n}\sum_{t=1}^{n} d_t^4.$$

From these:
- Skewness (asymmetry):
$$\gamma_1 = \frac{m_3}{\sigma^3} \quad (\text{when } \sigma^2>0).$$
- Kurtosis (tailedness; non‑excess):
$$\kappa = \frac{m_4}{\sigma^4} \quad (\text{when } \sigma^2>0).$$
For a normal distribution, $\gamma_1 \approx 0$ and $\kappa \approx 3$.

The bimodality coefficient (BC) is:
$$\text{BC} = \frac{\gamma_1^2 + 1}{\kappa}.$$

Reference points:
- Normal‑like: $\gamma_1 \approx 0$, $\kappa \approx 3 \Rightarrow \text{BC} \approx \tfrac{1}{3}$.
- Uniform distribution: $\kappa \approx 1.8 \Rightarrow \text{BC} \approx \tfrac{1}{1.8} \approx 0.555$.
- Many clearly bimodal distributions yield $\text{BC} > 0.555$.

Numerical stability note: if $\sigma^2 \approx 0$, we set $\gamma_1=0$, $\kappa=3$ to avoid division by very small numbers (this corresponds to a degenerate distribution).

---

## Decision rule (defaults)

We require a minimum number of completion tokens to have a reliable signal:
- $n \ge 8$. Otherwise we report "insufficient evidence" and do not reject.

If $n \ge 8$, we flag as suspicious (reject) when either condition holds:
- Unimodal‑low: $\text{median}(p) \le 0.20$.
- Bimodal: $q_{10}(p) \le 0.20$ and $\text{BC} \ge 0.58$.

Why this works:
- Same‑model sampling tends to have high median and BC below the bimodality threshold.
- Unimodal‑low (validator finds many tokens unlikely) is caught by a low median.
- We only trust BC when there is a genuine low tail (gated by $q_{10}$) to avoid false positives on unimodal‑high outputs.

---

## The algorithm (validator-side)

1. Run the validator model on the full sequence to get logits for every position.
2. For completion tokens (indices $t=\text{prompt\_length},\dots$), take predictive logits at $t-1$,
   convert to probabilities with softmax, and record $p_t$ for the chosen token.
3. Compute summary statistics: median(p), q10(p), $\gamma_1$ (skewness), $\kappa$ (kurtosis), and $\text{BC}$. Optionally also compute `low_frac`, `high_frac`, `mid_frac` as descriptive stats.
4. If $n < 8$: return "insufficient evidence".
5. Otherwise, apply the thresholds above. If any condition holds, reject as suspicious; else pass.

Performance: one forward pass; no extra dependencies (NumPy is optional for speed).

---

## Monitoring and logging

For easier debugging, metrics are logged to Weights & Biases under the `sampling_shape_check.*` namespace:

- `sampling_shape_check.hist`: histogram of chosen-token probabilities
- `sampling_shape_check.mean`, `sampling_shape_check.median`, `sampling_shape_check.q10`
- `sampling_shape_check.low_frac`, `sampling_shape_check.high_frac`, `sampling_shape_check.mid_frac`
- `sampling_shape_check.bc`, `sampling_shape_check.n`

These complement the decision rule (median and q10-gated BC) and help tune thresholds if needed.

---

## Worked examples

- Same‑model sampling (healthy):
  - $\text{low\_frac} \approx 0.05$, $\text{high\_frac} \approx 0.70$,
    $\text{mid\_frac} \approx 0.25$, $\text{BC} \approx 0.45$ → pass.

- Mismatched sampling (suspicious):
  - $\text{low\_frac} \approx 0.30$, $\text{high\_frac} \approx 0.55$,
    $\text{mid\_frac} \approx 0.15$, $\text{BC} \approx 0.62$ → fail.

---

## Edge cases and tips

- Very short completions lack signal; we skip with "insufficient evidence".
- Strong decoding constraints (e.g., low temperature, aggressive top‑k/top‑p) can increase
  $\text{high\_frac}$, but by themselves should not make $\text{low\_frac}$ large and
  $\text{mid\_frac}$ very small together with high BC.
- Deterministic decoding (greedy) often yields very high $\text{high\_frac}$; the test relies
  on the combination with a large low tail and high BC to avoid false positives.

---

## Configuration (environment variables)

- `GRAIL_SAMPLING_MIN_STEPS` (default 8)
- `GRAIL_SAMPLING_BC_THRESHOLD` (default 0.58)
- `GRAIL_SAMPLING_MEDIAN_LOW_MAX` (default 0.20)

If unset, defaults are used.

---

## Implementation details

The check is implemented in `grail/grail.py` in the `Verifier` class:

- `_collect_chosen_token_probs()`: Extracts validator probabilities for chosen tokens
- `_bimodality_metrics()`: Computes median, q10, skewness, kurtosis, BC (and descriptive fractions)
- `_token_sampling_shape_check()`: Applies the simplified decision rule and returns (pass/fail, metrics)

The check runs automatically during rollout verification, after GRAIL proof validation and termination checks but before solution validation.
