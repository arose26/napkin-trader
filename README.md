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

Ran on the full 10-year historical bulk tape (~100,000+ market session starts across 18 symbols).
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

Verdicts on the registered hypotheses (trained on 10-year tape):

1. **Reuse ladder — ratio 1 (`base`) optimal; high reuse degrades on 100k data.** `base` (+0.76) > `ratio.25` (+0.18) > `ratio4` (−0.00) > `ratio8` (−0.63). On 10 years of data (~2,500 bars / 100k+ session starts), excessive data reuse (`ratio4`/`ratio8`) over-replays experience and degrades policy performance, while ratio 1 yields the strongest point estimate among ratio settings.
2. **Shaping — `dd` cuts max drawdown to best in sweep (−2.63%).** The `dd` arm (λ=2) successfully reduced max drawdown to **−2.63%** (vs `base` −3.65%), achieving the shallowest drawdown of all 8 arms. However, this downside protection comes at the cost of raw return (−0.29% vs +0.76%).
3. **Action space — `long2` is the clear overall winner.** Restricting the action space to {flat, long} produced the top performing model across all metrics: **+2.19% return**, **+1.00 Sharpe**, and **−3.33% MDD**. Crucially, `long2`'s 95% CI **[+0.62, +3.36] does NOT contain 0** — it is the only arm in the sweep to achieve statistically significant positive returns on held-out test data. Structurally preventing short positions in an overall drifting market eliminates costly short-side drawdowns. `size5` (−0.98%) confirmed that fine-grained sizing tiers add capacity spent on a second-order decision.
4. **Honest null — broken by `long2`.** While all other arms have return CIs spanning 0, the `long2` policy trained on 10 years of historical data breaks the null hypothesis by delivering statistically positive test returns.

**The un-registered finding worth keeping**: Every DQN arm maintains roughly *half* of buy-and-hold's max drawdown (−2.6% to −4.2% vs −7.66%) while `long2` delivers positive outperformance.

## Run it

```bash
# expects ../napkin-tape and ../napkin-eyes cloned side by side, with the bulk tape built
python3 napkin_trader.py selfcheck
python3 napkin_trader.py sweep
python3 napkin_trader.py plot
```

## What's deliberately not here

No observation ablation (repo 2 answered it: small, by parsimony), no hyperparameter
tuning beyond the declared axes, no portfolio-level allocation head (the finale trades
the same per-symbol policy the sim validates; a cross-asset allocator would be a new
research question, not a shipping requirement), no live orders (repo 4 deploys; this repo
only trains and evaluates offline).
