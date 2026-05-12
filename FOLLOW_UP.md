## Extended Approach: SFT Warmup + GRPO Refinement

### Dataset preprocessing

Rather than relying solely on prompt engineering to establish the three constraints,
I took a data-driven approach: deterministically transforming the GSM8K training set
so that every example simultaneously satisfies all three constraints by construction.

The transformation pipeline:

1. **Annotation stripping.** GSM8K reasoning steps contain inline calculation
   annotations of the form `<<48/2=24>>`. These were stripped entirely before
   digit substitution, producing clean English prose rather than
   `<<forty-eight/two=twenty-four>>` hybrids that teach the model a confusing
   mixed notation.

2. **Digit-to-word substitution.** Every remaining digit string in the reasoning
   was replaced with its English-word form via `num2words`, producing fully
   lipogram-compliant reasoning text.

3. **Answer line reformatting.** The GSM8K `#### <integer>` final answer was
   replaced with `### <integer-in-words>`, satisfying the format constraint.

The result: 7,473 training examples where every example is verified to be
format-compliant, lipogram-compliant, and math-correct by construction. This
addresses the cold-start problem that dominated the GRPO-only phase: instead of
asking RL to create constraint-compliant behaviour from a near-zero base rate,
SFT establishes it as the default policy before RL begins.

The prompt format was simplified to a minimal structure with no few-shot
demonstrations:
```python
Question:
<question text>
Answer:
```
This teaches the model the task structure via weights rather than context,
reducing inference-time prompt length from ~800 tokens (system prompt + 8
few-shots) to ~50 tokens.

---

### SFT warmup results

Training configuration: 3 epochs, `learning_rate=2e-5`, LoRA at rank 8 targeting
`q_proj, v_proj, up_proj, down_proj`, `completion_only_loss=True`
so gradient computed only on the answer tokens.

| Metric | Base | After SFT (3 epochs) |
|---|---|---|
| numeric correctness | 0 | 22 (11.0%) |
| no-digits compliance | 0 | 199 (99.5%) |
| format compliance | 0 | 175 (87.5%) |
| all three together | 0 | 22 (11.0%) |

The base column is zero across all metrics because the base model was evaluated
under the minimal `Question:/Answer:` prompt format it has never been trained for,
producing digit-laden free-form reasoning with no `###` marker. This is an honest
measurement of the evaluation protocol, not a model failure — the base model's
underlying GSM8K capability is ~30% under standard 8-shot CoT prompting.

Two observations from the SFT results that drove subsequent decisions:

**The bifurcation problem is solved.** In GRPO-only runs, lipogram and math
improved on different problem subsets — the policy either produced digit-free
wrong answers the correct answer with digits. After SFT, all-three equals math
exactly: every rollout that gets the math right also passes format and lipogram.
The model has internalised the constraints as default behaviour, not as a
competing objective.

**The math ceiling is visible.** Doubling epochs (6 total) produced only marginal
further math improvement, confirming that SFT is teaching surface-form mimicry
of arithmetic patterns, not arithmetic reasoning. The model learns to predict
`twenty-four` after `forty-eight/two=` as a language pattern; it does not learn
to divide. This is the structural limit of supervised fine-tuning at 0.5B scale
on a representation (word-form arithmetic) the model was not pretrained on.

---

### GRPO refinement attempt

With the SFT model establishing near-perfect constraint compliance, I attempted
to use GRPO to push math accuracy further. The reward was restructured to
concentrate gradient pressure on math correctness:

```python
score = (
    1.0 if format_pass else -2.0
    + 1.0 if lipogram_pass else -2.0
    + format_pass * lipogram_pass * 5.0 if math_correct else -1.0
)
```

The rationale: with format at 87.5% and lipogram at 99.5%, both constraint tiers
are near-saturated and have minimal within-group variance. Concentrating reward
weight on math (5.0×) ensures GRPO's within-group advantage computation is driven
by math-correctness differences rather than constraint differences. The strict gated nature is to ensure that the model doesn't abandon any of the previous learning with respect to the lipogram or format, and hence get's a small reward to do that

A reference policy KL penalty was used (`beta=0.04`) to prevent the GRPO gradient
from overwriting the SFT-learned constraint habits. Training ran for 300 steps at
`learning_rate=1e-6`, `temperature=0.7`, K=16 rollouts.

**Result: no measurable improvement in math after 300 steps.**

Diagnosis: `beta=0.04` was too high for the available math reward signal. The KL
penalty `−beta × KL(policy || reference)` fires on every step, while the math
reward signal (`r_cor = 5.0`) fires only on ~11% of rollouts. Once KL drift
exceeds roughly `r_cor_mean / beta = 0.44 / 0.04 ≈ 11` nats, the KL penalty
exactly cancels the math reward, and the optimizer stalls. This explains the flat
metrics across 300 steps despite healthy `reward_std`.

As an additional attempt, I reduced `betabeta=0.01  but even with valid gradient
signal, math improvement remained modest — consistent with the SFT finding that
the arithmetic ceiling is a model-capacity issue, not a training-recipe issue.

---

### Why math is fundamentally hard for this model

The 0.5B math ceiling is not a hyperparameter problem. Three structural reasons:

**Pattern matching, not computation.** LLMs predict next tokens from statistical
patterns in training data. `48 / 2 = 24` is memorised as a token-sequence
pattern, not computed. For word-form arithmetic (`forty-eight / two = twenty-four`),
the pattern is different from what the model was pretrained on. The model learns
the form of word-form arithmetic from SFT but cannot reliably execute it on novel
combinations.

**Word-form arithmetic is harder than digit-form.** The model's arithmetic
knowledge is stored in digit-token-pattern form from pretraining. Evaluating
`forty-eight / two` requires an implicit cross-representation translation that
adds a reasoning step the model doesn't reliably execute. Rollout inspection
confirmed this: trained outputs like `one hundred - six = twelve` (should be
ninety-four) show syntactically correct structure with semantically wrong
arithmetic — the form was learned, the computation was not.

**The 0.5B capacity threshold.** Chain-of-thought reasoning provides marginal
benefit below roughly 7B parameters because the model lacks capacity to reliably
attend to and integrate intermediate results. Published GSM8K results confirm this:
Qwen2.5-0.5B achieves ~30% under relaxed evaluation, while Qwen2.5-7B achieves
~85%. The gap is not purely training — it reflects a genuine capacity threshold
for arithmetic execution. RLVR cannot create arithmetic capability; it can only
reinforce patterns already present in the policy's sampling distribution.

---

### What this demonstrates

The SFT + GRPO pipeline successfully solved the constraint-learning problem.
Format and lipogram compliance reached near-saturation (87.5% and 99.5%) and
the bifurcation failure mode was eliminated. The remaining gap — 89% failure rate
on the all-three conjunction — is attributable entirely to math errors, not
constraint violations.

This isolates math accuracy as the binding constraint at 0.5B scale under the
lipogram. The two-stage recipe (SFT for constraints, RL for capability) is the
correct approach; the limit is model capacity, not recipe design. With a 7B+
base model — where word-form arithmetic is more reliable and CoT works as
intended — the same pipeline would be expected to produce substantially higher
all-three accuracy. This is the specific change I would make with H100 compute
and a 24-hour budget.