# v1-vs-v3 mixture at 4B and 9B — the gain tracks where v1 failed

> 2026-08-06, eval box; **§1 added 2026-08-07** when the 4B v3 50k fork published.
> Companion to `mixture_v1_vs_v3_08b.md` (budget-matched at 50k).
> **4B is now budget-matched too (§1) — that is the headline.** §2 keeps the original
> 25k-vs-30k tables, which remain the source of the half-budget efficiency claim and of v1's
> own 30k→50k trajectory. 9B (§3) is STILL 25k-only, so its primary comparison is against
> v1's `step30000-swa` — the nearest revision, whose extra 20% of tokens runs AGAINST v3,
> making that win conservative; v1 `step50000-swa` there mixes mixture with budget and is an
> EFFICIENCY statement, not a clean A/B.
> Decode identical throughout: full canvas, block=32, steps=32, low-confidence, greedy,
> bs=1; gen=512 code / 1024 GSM8K. Revision is in every filename and `decode` block.
> Dumps in `_runs/battery4b/`.

## 1. 4B — BUDGET-MATCHED (both 50k), the definitive comparison

Added 2026-08-07, once the 4B v3 50k fork published. Identical token budget, mixture is the
only deliberate difference. **This supersedes the 25k-vs-30k table below as the headline.**

| benchmark | v1 @50k | v3 @50k | delta | McNemar p |
|---|--:|--:|--:|--:|
| MMLU 5-shot | 41.81 | **58.00** | **+16.19** | ≪0.001 |
| GSM8K strict | 45.94 | **54.97** | **+9.02** | 8e-11 |
| GSM8K flexible | 46.32 | **55.42** | **+9.10** | 4e-11 |
| MBPP `[BEGIN]` | 41.40 | **49.80** | **+8.40** | <0.0001 |
| MBPP fence | 43.00 | **50.00** | **+7.00** | 0.0003 |
| HumanEval | 57.93 | **59.76** | +1.83 | 0.75 n.s. |

**Four of five significant, none negative, at equal budget.** HumanEval is honest parity, not
a win (21 vs 18 discordant). The 25k worry that v1 might genuinely lead on HumanEval
(v3 was −3.05 there) is resolved: v3 gains +4.88 across its own second half and draws level.

### ⚠ CORRECTION: v3 does NOT "hold" knowledge — it trades more shallowly

An earlier framing in this doc predicted v3 would retain knowledge through the decay phase
where v1 collapsed. It does not. Each mixture's OWN second half:

```
v3, 25k -> 50k :  MMLU -4.41   GSM8K -3.10   HumanEval +4.88   MBPP +2.60   fence +2.00
v1, 30k -> 50k :  MMLU -6.05   GSM8K -2.35   HumanEval +11.59  MBPP +1.40
```

**Both mixtures trade knowledge AND math for code during the decay phase.** This is a property
of the recipe, not a v1 pathology. v3's trade is shallower and starts from far higher, which is
why it ends +16 MMLU while matching on HumanEval. State it that way; "v3 avoids the trade" is
not supported.

### Release-checkpoint consequence (open decision)

| | v3 @25k | v3 @50k |
|---|--:|--:|
| MMLU | **62.41** | 58.00 |
| GSM8K strict | **58.07** | 54.97 |
| HumanEval | 54.88 | **59.76** |
| MBPP | 47.20 | **49.80** |

25k is the better knowledge/math endpoint, 50k the better code endpoint. Which to release is
now an empirical question, and the 9B 50k fork will pose the same choice — worth deciding the
policy before it lands rather than after.

## 2. 4B at 25k vs 30k (superseded as headline; kept for the half-budget claim)

| benchmark | v3 @25k | v1 @30k | v1 @50k | v3−v1@30k | p | v3−v1@50k | p |
|---|--:|--:|--:|--:|--:|--:|--:|
| MMLU 5-shot | **62.41** | 47.86 | 41.81 | **+14.56** | ≪0.001 | **+20.60** | ≪0.001 |
| HumanEval | **54.88** | 46.34 | 57.93 | **+8.54** | 0.029 | −3.05 | 0.50 n.s. |
| MBPP `[BEGIN]` | **47.20** | 40.00 | 41.40 | **+7.20** | 0.00016 | **+5.80** | 0.0035 |
| MBPP fence | **48.00** | 41.40 | 43.00 | **+6.60** | 0.0011 | **+5.00** | 0.013 |
| GSM8K strict | **58.07** | 48.29 | 45.94 | **+9.78** | 4e-12 | **+12.13** | 1e-17 |
| GSM8K flex | **58.53** | 48.75 | 46.32 | **+9.78** | 2e-12 | **+12.21** | 1e-17 |

**Clean sweep: v3@25k beats v1@30k on all five benchmarks, every one significant**, while
holding 17% fewer tokens. Math is the second-largest gain after knowledge — notable because
v3 has slightly LESS math (15% vs v1's 16%); the gain comes from general web + knowledge-QA
supporting multi-step reasoning that v1's code saturation was crowding out.

Against the budget-comparable arm v3 wins **everything, significantly, while holding 17%
fewer tokens.** Against the released 50k artifact it wins MMLU by +20.60 and MBPP by +5.80
on **half** the budget, and is statistically indistinguishable on HumanEval.

### The smoking gun: v1's own 30k→50k trajectory

| v1 @4B, 30k → 50k | delta | |
|---|--:|---|
| HumanEval | **+11.59** | |
| MBPP `[BEGIN]` | +1.40 | |
| **GSM8K strict** | **−2.35** | p=0.09, suggestive |
| **MMLU** | **−6.05** | not in doubt |

**v1's MMLU and GSM8K both go DOWN with more training.** Those last 20k steps on the
code-heavy v1 mixture bought +11.6 HumanEval and paid −6.1 MMLU *and* −2.4 GSM8K. That is
the code-vs-everything-else trade made directly visible rather than inferred — and it was
getting worse the longer v1 ran.

⚠ An earlier version of this paragraph said "v3 does not make the trade". **That is wrong**
— §1 shows v3 makes the same trade, just shallower (MMLU −4.41 / GSM8K −3.10 across its own
second half) and from a much higher starting point. The correct contrast is depth-of-trade
and starting position, not presence-vs-absence.

## 3. 9B — v3 exceeds the AR model it was converted from

| benchmark | v3 @25k | v1 @30k | v1 @50k (released) | v3−v1@30k | p |
|---|--:|--:|--:|--:|--:|
| MMLU 5-shot | **74.68** | 73.09 | 71.87 | +1.59 | **0.0024** |
| HumanEval | **63.41** | 59.76 | 64.63 | +3.65 | 0.39 n.s. |
| MBPP `[BEGIN]` | **51.20** | 50.00 | 54.00 | +1.20 | 0.59 n.s. |

**The honest 9B read: the mixture buys KNOWLEDGE and is NEUTRAL on code.** All three deltas
are positive but only MMLU is significant; MBPP's 47-vs-41 discordant split is what noise
looks like. This is a much narrower result than 4B's five-benchmark sweep, and it is exactly
what the conversion-gap picture predicts — 9B v1 had already reached its AR twin, so there
was nothing to rescue. Against the released v1@50k on half the tokens: MMLU +2.81 but
HumanEval −1.22 and MBPP −2.80.

**Consequence for the 9B 50k fork:** the argument for v3 at 9B is "same code, better
knowledge, reached in half the steps" — NOT "better at everything". The 4B result cannot be
used to justify the 9B fork; the sizes demonstrably behave differently. 9B GSM8K is the
missing datapoint most likely to separate the arms (math was 4B's second-largest gain) and
has NOT been run.

The family's standing claim was *"at 9B the conversion is essentially lossless"* — 71.87 vs
the Qwen3.5-9B-Base AR twin's 71.93. **v3 @25k scores 74.68, i.e. +2.75 ABOVE its AR twin, on
half the training budget.** A masked diffusion LM beating the autoregressive model it was
initialised from is a materially stronger claim than lossless conversion.

⚠ **Two caveats before this is quoted.** (a) The AR twin baseline is itself flagged in
`mmlu_v1.md`: Qwen3.5-4B (73.02) ≥ Qwen3.5-9B (71.93) is an unusual scaling curve and that doc
already says to re-measure the 9B AR baseline before publication. The +2.75 rests entirely on
71.93 — **re-running that AR baseline is the highest-value single run available** and it is
cheap (stock lm-eval `hf`, zero dqeval code). (b) v3@25k vs v1@50k confounds mixture with
budget; against the budget-comparable v1@30k the gain is +1.59, not +2.81.

## 4. The cross-scale pattern — gain tracks v1's deficit, not size

MMLU, v3 vs the nearest v1 arm:

| size | v3 | v1 | delta | note |
|---|--:|--:|--:|---|
| 0.8B | 30.22 | 26.46 | +3.76 | budget-matched @50k |
| **4B** | 62.41 | 47.86 | **+14.56** | v1 has +20% tokens |
| 9B | 74.68 | 73.09 | +1.59 | v1 has +20% tokens |

Non-monotonic, and the shape is explained by the **conversion gap to the AR twin** rather
than by scale:

| size | AR twin | v1 gap (released 50k) | v3 gap | closed by |
|---|--:|--:|--:|--:|
| 0.8B | 48.36 | −21.90 | −18.14 | 3.76 |
| 4B | 73.02 | −31.21 | **−10.61** | **20.60** |
| 9B | 71.93 | −0.06 | **+2.75** | 2.81 |

**The mixture is a win at every scale, but its magnitude is a function of how badly v1 was
failing at that size.** 4B was v1's worst retention failure and gets the largest rescue; 9B
had already retained, leaving little headroom — v3 still wins, significantly, but by 1.6 not
14.6. So this is NOT merely a small-model rescue (9B gains too, and clears its AR twin), and
it is NOT a uniform constant either.

## 5. Practical consequence for the campaign
The assumed trade — *"sacrifice a little coding performance at 4B/9B for large general
gains"* — **does not appear to exist.** At 4B v3 wins code and knowledge simultaneously
against the budget-comparable arm. The one place v1 still leads is 4B HumanEval at full 50k
budget (57.93 vs 54.88, n.s.), which is exactly the metric v1 bought with its −6.05 MMLU.
