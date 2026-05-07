# Lord of the Strings — Writeup

**Model:** Qwen/Qwen2.5-0.5B-Instruct, LoRA only
**Algorithm:** GRPO via `trl.GRPOTrainer`, no reference policy (`beta=0`)
**LoRA targets:** `q,k,v,o,up,down_proj` at rank 8, alpha 16
**Compute:** Google Colab - Tesla T4 16GB
**Wall clock time:** ~ 3 hours 50 mins

Guiding principle: **RLVR cannot create behaviour, only amplify it.** The base policy emits digit-free output ~0.7% of the time at T=1.0. Every choice below — reward shape, K, LoRA targets, prompt — is downstream of that.

---

## 1. Reward design

The reward is an inclusion–exclusion sum: every non-empty subset of the three indicators contributes a term.

```python
score = (
    r_fmt + r_lip + r_cor                              # singles, weight 1
    + 2.0 * (r_fmt*r_lip + r_lip*r_cor + r_cor*r_fmt)  # pairs,   weight 2
    + 3.0 * r_fmt*r_lip*r_cor                          # triple,  weight 3
)
```

`r_fmt` requires a `### <answer>` final line and a parseable integer (digit or word form). `r_cor` is binary correctness against gold. `r_lip` is `max(0, 1 - 0.25 * digit_count)`. Payoffs at the binary corners: one tier scores 1, two tiers score 4, all three score 12.

Chosen by elimination:

1. Plain weighted sum: policy converges on whichever tier is cheapest. Base format compliance was already 160/200 under the few-shot prompt — format is nearly free. A flat sum collapses to "always emit `### <random word>`" without addressing the harder tiers.
2. Pure product: zeros out almost every rollout in early training. Within-group reward std collapses, GRPO's advantage vanishes.
3. Inclusion–exclusion solves both: each tier earns standalone reward (preserves cold-start signal); pairs are worth 2× a single tier; the triple is meaningful (3× a pair) but not crushing, so partial paths stay competitive.

A note on `r_lip` linearity: the 0.25 slope only ramps over `digit_count ∈ {0, 1, 2, 3}`. For ≥4 digits, `r_lip` floors at 0 — and most untrained rollouts emit ≥4 digits, since GSM8K is digit-heavy. So `r_lip` is effectively a soft binary; a strict binary would behave similarly. The RL signal operates mostly on the lipogram tier, with format and correctness as stabilisers.

---

## 2. Parameters

### 2.1 GRPO config

| Hyperparameter | Value | Defense |
|---|---|---|
| `temperature` | 1.0 | Calibrated against measured base diversity; prevents collapse. |
| `top_p` | 0.95 | Balances probable tokens against exploration. |
| `max_completion_length` | 512 | GSM8K reasoning fits in <300 tokens; comfortable headroom. |
| `num_generations` (K) | 16 | Justified empirically below. |

I sampled 32 rollouts × 200 prompts at T=1.0 on the base model and counted digit-free completions: roughly 50 / 6,400 ≈ 0.7% per-rollout. The probability that any rollout in a group of K is digit-free is `1 − (1 − 0.007)^K`. K=4 yields 2.8%, K=8 yields 5.5%, K=16 yields 10.7%, K=32 yields 20.3%.

Within-group variance on `r_lip` is *necessary but not sufficient* — if the digit-free rollout has poor math, total advantage may push *away* from digit-free. So the K threshold isn't binary "learns/doesn't"; it's "how often the gradient on `r_lip` is informative." At K=8 (~1 group in 18), signal was too sparse for the cosine-decayed schedule. K=16 (~1 group in 9) empirically passed. K=32 was infeasible on T4 once gradient accumulation and LoRA optimiser state were accounted for.

Eval uses `do_sample=False` for both base and trained — capability measurement should be deterministic. Training samples; eval is greedy. The asymmetry is intentional.

### 2.2 LoRA config

| Hyperparameter | Value | Defense |
|---|---|---|
| rank `r` | 8 | Sufficient for a stylistic-shift task. |
| alpha | 16 | `alpha = 2 * r`, the standard scaling. |
| target modules | attention (q,v) + MLP (gate, up and down) | Attention shapes output mode; MLP shapes the token-level vocabulary needed for the lipogram. |
| dropout | 0.05 | Light regularisation against trajectory collapse under sparse rewards. |

The principle: adapt parameters that govern *behaviour*, leave *factual knowledge* alone, use the smallest rank that supports the task.

Rank 8 is sufficient. The lipogram is a stylistic constraint layered on math the base model already does ~30% on. Higher ranks under noisy reward let additional degrees of freedom rewrite useful base capabilities (math) faster than the policy reinforces new ones (lipogram). Rank 8 produces ~9M trainable parameters.

Module choice spans both attention and MLP. I started with attention-only (`q,k,v,o`), reasoning that attention controls "how the model writes" while MLPs store "what it knows." Subjective inspection of that run showed format compliance came up but lipogram appeared to plateau — the model did not effectively override its digit-based output prior at the vocabulary level. Adding `up_proj` and `down_proj` gave the policy access to MLPs, which write directly to the residual stream just before the next-token logits. I do not have logged eval numbers from the attention-only run.

On the math regression: trained accuracy dropped from 19.5% (base) to 11%. I cannot, with the runs I have, fully disambiguate the cause. Candidates: (a) undertraining; (b) KL drift in the absence of a reference policy; (c) lipogram-driven distribution shift away from digit-anchored chain-of-thought patterns; (d) a structural limit at 0.5B for prose-form arithmetic. Evidence for (c)/(d): base accuracy under the symbolic-anchored prompt was the highest measured (19.5%), and forcing word-form reasoning removes a scaffold the base model relies on. I did not run the controlled ablation that would separate (c) from (d) — calling this purely "structural" would be overclaiming.

---

## 3. Failure modes I hit and fixed

### 3.1 Verifier robustness: `word2number` IndexError

At a random step of an earlier run, training crashed:

```
File ".../word2number/w2n.py", in word_to_num
    ... = numbers[0]
IndexError: list index out of range
# Repro: w2n.word_to_num("thousand five")
```

`word2number` parses thousand-multiplier patterns by computing the multiplier from preceding tokens. With `thousand` at position 0 and number-words after, the slice is empty and `numbers[0]` raises `IndexError` — *not* `ValueError`, which my verifier was catching, so it propagated.

The bug is broader. Same path triggers for `million` and `billion`. Separate silent-wrong case: `w2n.word_to_num("hundred and one")` returns 100, because the bare `hundred` is parsed as the whole expression.

Fix: a pre-flight check that prepends `"one"` whenever the input starts with a magnitude word. Handles both crash and silent-wrong cases uniformly. The verifier wraps the call in `except Exception` as belt-and-braces.

Lesson: RLVR verifiers must be robust against arbitrary policy output, including outputs that expose latent bugs in third-party parsers.

### 3.2 GRPO variance starvation from low K

Early on I reduced `num_generations` from 16 to 4 to save memory. Training reward flatlined at exactly 2.0 with `reward_std ≈ 0` for tens of consecutive steps. Reward function was correct — manually scoring rollouts produced varied rewards. Within each group of 4, the policy was producing essentially identical completions, all scoring the same partial-credit value.

GRPO's advantage is `(r − group_mean) / group_std`. When all rollouts score the same, std is zero, advantage is zero, gradient is zero. Reward shape is irrelevant once within-group variance collapses.

Cause: low K gave too few independent samples, and lowering temperature (which I'd considered to make completions "cleaner") would have made it worse. Minimum viable was K=16 at T=1.0.

Lesson: in GRPO, within-group variance is a hyperparameter as fundamental as the reward shape.

---

## 4. What I would change with a 24-hour H100

The configured schedule was 200 optimizer steps, but Colab disconnected at step ~50, so the LR never decayed below ~75% of peak — the cosine schedule's design point was never reached. Several items below are blocked not just by scale but by that truncated horizon.

**SFT warmup.** Biggest leverage point. RLVR amplifies behaviour, it does not create it. With H100 budget I would generate 200–500 digit-free GSM8K-style solutions, run one epoch of supervised LoRA, then start GRPO. Raises the base rate enough that the lipogram tier receives signal from step 0.

**Larger base model.** Qwen2.5-0.5B is below the threshold where prose-form arithmetic is reliable. Qwen2.5-3B-Instruct should absorb both constraints without paying the math/lipogram trade-off as steeply.

**vLLM-backed K=64+ sampling.** On T4 I was capped at K=16 by memory. Higher K maintains within-group variance even when individual tiers approach saturation, which becomes critical late in training when the bottleneck shifts to the rare conjunction.

**Curriculum reward.** Phase 1 (0–1k steps): format + lipogram. Phase 2 (1k–3k): add standalone math. Phase 3 (3k+): joint bonus dominates. Curriculum requires a longer horizon than I had — at ~50 steps, all phases would have collapsed into Phase 1.

**Generalisation note.** The within-group-variance failure mode generalises beyond LLMs: any group-relative RL on sparse-reward tasks (including token-level robotic control) suffers the same collapse. The calibration → K-derivation procedure transfers directly.

---

## Disclosures

- **No external LLM calls** in the reward, eval, or verifier. Verifier is deterministic regex + `word2number` with explicit handling for known `w2n` quirks (the `IndexError` family at multiplier-keyword position 0; the silent-wrong "hundred and one" → 100 case). Both reproduced and fixed.
- **Detailed diagnosis of further cases** trimmed for the word limit. Happy to present them if possible.
