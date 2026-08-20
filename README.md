# napkin-trader

Repo 3 of the **napkin-trader series** ([napkin-tape](https://github.com/arose26/napkin-tape)
→ [napkin-eyes](https://github.com/arose26/napkin-eyes) → this → napkin-gap →
napkin-wallstreet). Repo 1 built the calibrated replay sim; repo 2 showed observation
choice is second-order and everything ties at this data scale. This repo trains the agents
that will actually trade on [ClawStreet](https://www.clawstreet.io), and asks the two
questions the finale depends on:

> **Does the series-1 lesson — data reuse is the whole game — survive the jump from
> MinAtar to financial tape? And what does risk-aware reward shaping actually buy,
> measured in drawdown and Sharpe, not vibes?**

## Setup

Machinery imported from napkin-eyes (env validated against napkin-tape's exact reference
sim to 0.079%; walk-forward splits train < 2026-01-01, val < 2026-06-01, test = the rest;
same frozen DQN recipe: buffer + target net + 3-step, no double-Q). Base config: `raw10`
observations (repo 2's parsimony carry-forward), actions {short, flat, long}, raw log-PnL
reward, replay ratio 1. Each arm changes exactly one thing:

| axis | arm | change |
|---|---|---|
| — | `base` | as above |
| reward | `dd` | drawdown-penalized: r − λ·(new drawdown depth), λ=2 |
| reward | `sharpe` | variance-penalized (Sharpe proxy): r − κ·r², κ=10 |
| actions | `long2` | {flat, long} — no shorting |
| actions | `size5` | {−1, −½, 0, +½, +1} — size tiers |
| ratio | `ratio.25` | 1 gradient batch per 4 batches of experience |
| ratio | `ratio4` | 4× reuse |
| ratio | `ratio8` | 8× reuse |

10 seeds per arm (80 runs), IQM + 95% bootstrap CI, ties reported as ties. Test metrics
per run: **total return, daily Sharpe (annualized ×√252), max drawdown** — the shaping
arms' claim is about the risk columns, not the return column.

## Hypotheses (registered 2026-08-19, before the sweep ran)

1. **Reuse still dominates, in-domain.** The train set is ~10k distinct states — reuse
   is the only way to learn it. Predict: `ratio4` ≥ `base` > `ratio.25` on test return
   point estimates, with `ratio8` at or below `ratio4` (the knee, echoing napkin-replay's
   MinAtar result). Given repo 2's variance lesson, we expect CIs to overlap — the
   registered claim is about the *point-estimate ordering* of the ratio ladder, and we
   will report it as weak evidence unless CIs separate.
2. **Shaping buys risk, costs nothing measurable in return.** `dd` and `sharpe` arms cut
   max-drawdown IQM vs `base` (this is the one comparison we predict escapes the tie
   zone) and improve or tie Sharpe; raw return ties. This pair becomes repo 5's
   board-chaser (`base`) vs flagship (`dd` or `sharpe`, whichever wins the risk columns).
3. **Action space ties on return.** `long2` ≈ `base` (shorting a mostly-rising tape adds
   a way to lose, but the agent can learn "don't"); `size5` ties (finer sizing is capacity
   spent on a second-order decision). `long2` should show *shallower drawdowns* than
   `base` for free (it structurally can't be short in a rally).
4. **The honest null, again:** no arm significantly beats same-window buy-and-hold on raw
   return; every return CI contains 0. Alpha is not the claim; risk shape and the reuse
   ladder are.

## Selfchecks (`python3 napkin_trader.py selfcheck`)

- **The substitution trick** (the series rule for trained policies): extract a net's
  greedy actions on the test window, then replay those exact actions through
  napkin-tape's independently-coded CPU accounting — equity curves must agree.
- **Reward identities**: `base` cumulative reward ≡ log final equity; `dd` penalty
  ≡ Σ new-drawdown increments (recomputed from the equity curve after the fact);
  `sharpe` penalty ≡ κ·Σr².
- **Action-space honesty**: a `long2` policy can never hold a short (asserted over a
  full eval); `size5` positions only ever take the 5 declared values.
- **Ratio bookkeeping**: gradient samples consumed ≡ ratio × transitions added (±1 batch).

## Results

Trained on the aligned daily panel of 18 symbols, **1,298 bars (2021-06-17 → 2026-08-19)**, at
~1.02M sampled transitions per run (256 envs × 4,000 vectorized steps). The bars figure deserves
its own note: ten years sit on disk (2,512 stock bars from 2016-08-22), but the tape aligner kept
only dates present for *every* symbol, and `X:SOLUSD` lists on 2021-06-17 — so one late listing
was discarding **48% of the history**. napkin-eyes now offers `align="ragged"` (the stock calendar,
late listers masked before their first bar) which recovers all 2,512 bars; the numbers below
predate it and are on the 1,298-bar panel.
Test window 2026-06-01 → 2026-08-18 (54 bars); same-window buy-and-hold: −0.27% return, −7.66% max drawdown.

![results](assets/hero.png)

| arm | return IQM [95% CI] | Sharpe IQM | MDD IQM [95% CI] |
|---|---|---|---|
| base | +0.76 [−1.83, +3.11] | +0.18 | −3.65 [−5.17, −3.22] |
| dd | −0.29 [−1.26, +0.53] | −0.22 | −2.63 [−3.32, −2.20] |
| sharpe | −0.66 [−2.01, +0.75] | −0.46 | −3.23 [−3.72, −2.88] |
| long2 | +2.19 [+0.62, +3.36] | +1.00 | −3.33 [−3.92, −2.77] |
| size5 | −0.98 [−3.33, +1.73] | −0.68 | −3.58 [−5.37, −2.62] |
| ratio.25 | +0.18 [−1.51, +2.19] | +0.08 | −4.07 [−5.03, −3.17] |
| ratio4 | −0.00 [−2.55, +2.33] | +0.01 | −4.20 [−6.12, −2.73] |
| ratio8 | −0.63 [−2.11, +1.05] | −0.36 | −3.93 [−5.32, −2.83] |

Verdicts on the registered hypotheses (1,298-bar panel, single 55-bar test window):

1. **Reuse ladder — point ordering favours ratio 1, but every CI overlaps.** `base` (+0.76) > `ratio.25` (+0.18) > `ratio4` (−0.00) > `ratio8` (−0.63), and no pair is separable: `base` [−1.83, +3.11] contains all three others' point estimates. The previous sweep's ordering was the *opposite* (`ratio4` > `base`), which is the strongest available evidence that this ladder's ordering is noise at this test size. No mechanism is claimed for it.
2. **Shaping — `dd` cuts max drawdown to best in sweep (−2.63%).** The `dd` arm (λ=2) successfully reduced max drawdown to **−2.63%** (vs `base` −3.65%), achieving the shallowest drawdown of all 8 arms. However, this downside protection comes at the cost of raw return (−0.29% vs +0.76%).
3. **Action space — `long2` is the best arm here, and its CI excludes 0.** Restricting the action space to {flat, long} gave the top point estimate on every metric: **+2.19% return**, **+1.00 Sharpe**, **−3.33% MDD**, with a 95% CI of **[+0.62, +3.36]** that does not contain 0. `size5` (−0.98%) suggests fine-grained sizing tiers spend capacity on a second-order decision.

   Three things that claim is **not**, stated because they are easy to assume:
   - **It is not "`long2` beats `base`".** That comparison was never run. `long2` [+0.62, +3.36] and `base` [−1.83, +3.11] overlap across most of `long2`'s range; a CI against *zero* and a CI against *another arm* are different tests. The `walkforward` subcommand exists to run the paired one.
   - **It is not corrected for multiplicity.** Eight arms were screened at 95%; the chance that at least one clears 0 by luck is ≈ 1 − 0.95⁸ ≈ **34%** if the arms were independent — lower given they share data and seeds, but nowhere near 5%.
   - **It is not fresh evidence.** The 55-bar test window is the same one used by the previous sweep and by napkin-eyes. Training data grew; the test evidence did not.
4. **Honest null — dented by `long2`, on one window.** Every other arm's return CI spans 0; `long2`'s does not. Whether that survives windows it was not selected on is the question `walkforward` answers, and it is the only version of this claim worth deploying against.

**The un-registered finding worth keeping**: every DQN arm holds roughly *half* of buy-and-hold's max drawdown (−2.6% to −4.2% vs −7.66%). Caveat carried from the previous sweep and still unresolved: the agents are only partially invested, so this is not yet an exposure-matched comparison.

Every number above is reproducible from `assets/sweep_results.json` (per-arm, per-seed metrics plus the bootstrap CIs).

### The multi-window check the single window couldn't do

`walkforward` splits the tail of the tape into **four contiguous, non-overlapping 55-bar test
windows**, each trained only on the bars before it, and reports the arms' **paired** difference per
window — the comparison "`long2`'s CI excludes 0" never makes. 2 arms × 4 folds × 10 seeds;
`selfcheck` asserts no fold trains past its own test start. Source: `assets/walkforward.json`.

| fold | window | `base` | `long2` | paired diff (`long2` − `base`) |
|---|---|---|---|---|
| 0 | 2025-10-02 → 2025-12-18 | +1.05 [−2.16, +3.78] | −0.58 [−1.92, +0.95] | **−1.29** [−3.32, +0.80] |
| 1 | 2025-12-19 → 2026-03-11 | −4.89 [−6.64, −2.32] | −5.54 [−6.58, −3.98] | **−0.72** [−2.63, +1.38] |
| 2 | 2026-03-12 → 2026-05-29 | −0.45 [−2.63, +2.38] | +4.01 [+1.42, +5.52] | **+4.03** [+0.69, +7.13] |
| 3 | 2026-06-01 → 2026-08-18 | −0.97 [−4.05, +2.08] | +0.97 [−1.17, +3.08] | **+1.51** [+0.13, +3.65] |

**`long2` beats `base` in 2 of 4 windows, and loses the other two.** The single-window headline does
not survive as a comparative claim. Three specifics:

- **The window dominates the arm.** `base` spans +1.05 to −4.89 across folds and `long2` −5.54 to
  +4.01. Fold 1 was negative for everything. Any single 55-bar verdict is mostly a statement about
  which quarter you happened to test on.
- **Fold 3 *is* the original test window, and it reads +0.97, not +2.19.** Same window, same arm —
  the only difference is that the fold trains through 2026-05-29 instead of stopping at
  2026-01-01. Adding 102 training bars more than halved the measured edge, which is the clearest
  available evidence that +2.19% was not a stable estimate.
- **What is left standing** is narrower and worth stating: in the two most recent windows,
  `long2`'s advantage over `base` is positive with a CI excluding 0. That is consistent with a
  long-only bias paying in a drifting market and losing when the drift stops (folds 0–1) — a
  regime-dependent edge, not a free one. Four windows cannot establish it.

**Consequence for deployment.** This is why the live board-chaser (NPKN, `base`) was *not* switched
to `long2` on the strength of the single-window result. `long2` already runs as the separately
disclosed flagship (NPKL), so the live pair tests the same contrast prospectively, which is the
only version of this comparison that isn't reusing a window.

## Run it

```bash
# expects ../napkin-tape and ../napkin-eyes cloned side by side, with the bulk tape built
pip install --target .deps "numpy<2"        # torch 2.2 needs numpy 1.x; without
                                            # this, selfcheck dies on dtype inference
PYTHONPATH=.deps python3.10 napkin_trader.py selfcheck
PYTHONPATH=.deps python3.10 napkin_trader.py sweep
PYTHONPATH=.deps python3.10 napkin_trader.py walkforward   # 4 windows, paired arms
PYTHONPATH=.deps python3.10 napkin_trader.py plot
```

## What's deliberately not here

No observation ablation (repo 2 answered it: small, by parsimony), no hyperparameter
tuning beyond the declared axes, no portfolio-level allocation head (the finale trades
the same per-symbol policy the sim validates; a cross-asset allocator would be a new
research question, not a shipping requirement), no live orders (repo 4 deploys; this repo
only trains and evaluates offline).
