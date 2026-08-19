#!/usr/bin/env python3
"""napkin-trader: the agent. Reward shaping x action spaces x replay ratio, one axis
at a time around a fixed base, on the frozen sim. Repo 3 of the napkin-trader series.

Commands: selfcheck | sweep [arm ...] | report | plot
Machinery imported from ../napkin-eyes (env validated vs napkin-tape's reference sim).
Registered hypotheses in README.md.
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "napkin-eyes"))
sys.path.insert(0, os.path.join(HERE, "..", "napkin-tape"))
import napkin_eyes as ne

OUT = os.path.join(HERE, "out")
DEV = ne.DEV
GAMMA, NSTEP, LR, BATCH = ne.GAMMA, ne.NSTEP, ne.LR, ne.BATCH
BUF_CAP, WARMUP, SYNC_EVERY = ne.BUF_CAP, ne.WARMUP, ne.SYNC_EVERY
N_ENVS, EP_LEN, VEC_STEPS = ne.N_ENVS, ne.EP_LEN, ne.VEC_STEPS
OBS_ARM = "raw10"                       # repo 2 carry-forward (parsimony)
LAMBDA_DD, KAPPA_SH = 2.0, 10.0
SEEDS = 10

ACTION_SETS = {"long2": [0.0, 1.0],
               "long3": [-1.0, 0.0, 1.0],
               "size5": [-1.0, -0.5, 0.0, 0.5, 1.0]}
ARMS = {  # arm -> (reward, actions, ratio); base = pnl / long3 / 1
    "base":     ("pnl", "long3", 1.0),
    "dd":       ("dd", "long3", 1.0),
    "sharpe":   ("sharpe", "long3", 1.0),
    "long2":    ("pnl", "long2", 1.0),
    "size5":    ("pnl", "size5", 1.0),
    "ratio.25": ("pnl", "long3", 0.25),
    "ratio4":   ("pnl", "long3", 4.0),
    "ratio8":   ("pnl", "long3", 8.0),
}


class QNet(nn.Module):
    def __init__(self, d, n_actions):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 128), nn.ReLU(),
                               nn.Linear(128, 128), nn.ReLU(),
                               nn.Linear(128, n_actions))

    def forward(self, x):
        return self.f(x)


def shape_reward(reward_kind, log_r, eq, peak):
    """Shaped training reward from the raw log-return step. eq/peak are the env's
    multiplicative equity and its running max AFTER applying this step."""
    if reward_kind == "pnl":
        return log_r
    if reward_kind == "sharpe":
        return log_r - KAPPA_SH * log_r ** 2
    if reward_kind == "dd":
        dd_new = 1.0 - eq / peak                       # depth in [0, 1)
        return log_r - LAMBDA_DD * torch.clamp(dd_new - shape_reward.prev_dd, min=0.0)
    raise SystemExit(reward_kind)


def train(arm, seed, market, feat, quiet=True, vec_steps=VEC_STEPS):
    reward_kind, aset, ratio = ARMS[arm]
    acts = torch.tensor(ACTION_SETS[aset], device=DEV)
    nA = len(acts)
    torch.manual_seed(seed)
    d = ne.obs_dim(OBS_ARM)
    net, tgt = QNet(d, nA).to(DEV), QNet(d, nA).to(DEV)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buf = ne.Replay(BUF_CAP, d, seed)
    nstep = ne.VecNStep(NSTEP, GAMMA, N_ENVS, d)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)

    def reset():
        s = torch.randint(0, market.S, (N_ENVS,), generator=g).to(DEV)
        t = torch.randint(61, market.t_train_end - EP_LEN - NSTEP,
                          (N_ENVS,), generator=g).to(DEV)
        return (s, t, torch.zeros(N_ENVS, device=DEV),
                torch.ones(N_ENVS, device=DEV), torch.ones(N_ENVS, device=DEV))

    sym, t, pos, eq, peak = reset()
    shape_reward.prev_dd = torch.zeros(N_ENVS, device=DEV)
    updates, added = 0, 0
    for step in range(vec_steps):
        o = ne.make_obs(feat, t, sym, pos, OBS_ARM)
        eps = 1.0 + min(1.0, step / (vec_steps * 0.3)) * (0.05 - 1.0)
        a = torch.where(torch.rand(N_ENVS, generator=g).to(DEV) < eps,
                        torch.randint(0, nA, (N_ENVS,), generator=g).to(DEV),
                        net(o).argmax(1))
        p_new = acts[a]
        f = market.step_factor(t, sym, pos, p_new)
        log_r = torch.log(f)
        eq = eq * f
        peak = torch.maximum(peak, eq)
        r = shape_reward(reward_kind, log_r, eq, peak)
        if reward_kind == "dd":
            shape_reward.prev_dd = 1.0 - eq / peak
        pos, t = p_new, t + 1
        o2 = ne.make_obs(feat, t, sym, pos, OBS_ARM)
        emitted = nstep.push(o, a, r, o2)
        if emitted:
            buf.add(*emitted); added += emitted[0].shape[0]
        if (step + 1) % EP_LEN == 0:
            for tup in nstep.flush(o2):
                buf.add(*tup); added += tup[0].shape[0]
            sym, t, pos, eq, peak = reset()
            shape_reward.prev_dd = torch.zeros(N_ENVS, device=DEV)
        while buf.n >= WARMUP and updates * BATCH < added * ratio:
            s_, a_, r_, s2_, m_ = buf.sample(BATCH)
            q = net(s_).gather(1, a_[:, None]).squeeze(1)
            with torch.no_grad():
                target = r_ + (GAMMA ** m_) * tgt(s2_).max(1).values
            loss = ((q - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            updates += 1
            if updates % SYNC_EVERY == 0:
                tgt.load_state_dict(net.state_dict())
    train.last_bookkeeping = (added, updates)
    return net, acts


shape_reward.prev_dd = torch.zeros(N_ENVS, device=DEV)


@torch.no_grad()
def evaluate(net, acts, market, feat, t0, t1):
    """Greedy rollout per symbol; returns (portfolio_curve, positions [T,S])."""
    net.eval()
    S = market.S
    sym = torch.arange(S, device=DEV)
    pos = torch.zeros(S, device=DEV)
    eq = torch.ones(S, device=DEV)
    curve, poss = [], []
    for t in range(t0, t1):
        o = ne.make_obs(feat, torch.full((S,), t, device=DEV), sym, pos, OBS_ARM)
        p_new = acts[net(o).argmax(1)]
        eq = eq * market.step_factor(torch.full((S,), t, device=DEV), sym, pos, p_new)
        pos = p_new
        poss.append(pos.tolist())
        curve.append(eq.mean().item())
    net.train()
    return curve, poss


def metrics(curve):
    c = np.array([1.0] + list(curve))
    r = np.diff(np.log(c))
    sharpe = float(r.mean() / (r.std() + 1e-12) * math.sqrt(252))
    peak = np.maximum.accumulate(c)
    mdd = float(((c - peak) / peak).min() * 100)
    return {"return_pct": (c[-1] - 1) * 100, "sharpe": sharpe, "max_drawdown_pct": mdd}


def sweep(arms=None, seeds=SEEDS):
    os.makedirs(os.path.join(OUT, "sweep"), exist_ok=True)
    market = ne.Market()
    feat = torch.tensor(ne.build_features(market, OBS_ARM), device=DEV)
    for arm in (arms or ARMS):
        for seed in range(seeds):
            fpath = os.path.join(OUT, "sweep", f"{arm}_{seed}.json")
            if os.path.exists(fpath):
                continue
            t0 = time.time()
            net, acts = train(arm, seed, market, feat)
            test, poss = evaluate(net, acts, market, feat, market.t_val_end, market.T)
            val, _ = evaluate(net, acts, market, feat, market.t_train_end, market.t_val_end)
            rec = {"arm": arm, "seed": seed, "test_curve": test,
                   "test": metrics(test), "val": metrics(val)}
            json.dump(rec, open(fpath, "w"))
            print(f"{arm} seed {seed}: test {rec['test']['return_pct']:+.2f}% "
                  f"sharpe {rec['test']['sharpe']:+.2f} mdd {rec['test']['max_drawdown_pct']:.1f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)
    report()


def report():
    import glob
    print(f"\n{'arm':9} {'n':>2} {'ret IQM':>8} {'sharpe IQM':>10} {'MDD IQM':>8}")
    for arm in ARMS:
        rs = [json.load(open(f)) for f in
              sorted(glob.glob(os.path.join(OUT, "sweep", f"{arm}_*.json")))]
        if not rs:
            continue
        cols = {k: ne.iqm([r["test"][k] for r in rs])
                for k in ("return_pct", "sharpe", "max_drawdown_pct")}
        print(f"{arm:9} {len(rs):>2} {cols['return_pct']:>+8.2f} "
              f"{cols['sharpe']:>+10.2f} {cols['max_drawdown_pct']:>8.2f}")


def plot():
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    market = ne.Market()
    c = np.exp(np.cumsum(market.logret, 0))
    bh_curve = (c[market.t_val_end + 1:] / c[market.t_val_end]).mean(1)
    bh = metrics(bh_curve.tolist())

    C = dict(zip(ARMS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                        "#e87ba4", "#008300", "#4a3aa7", "#e34948"]))
    INK, MUTED = "#1a1a19", "#6f6e64"
    stats = {}
    for arm in ARMS:
        rs = [json.load(open(f)) for f in
              sorted(glob.glob(os.path.join(OUT, "sweep", f"{arm}_*.json")))]
        if not rs:
            continue
        stats[arm] = {k: (ne.iqm([r["test"][k] for r in rs]),
                          *ne.bootstrap_ci([r["test"][k] for r in rs]))
                      for k in ("return_pct", "max_drawdown_pct")}
    arms_present = [a for a in ARMS if a in stats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color="#e6e5dc", lw=0.6)
        ax.set_axisbelow(True)

    arms = arms_present
    y = np.arange(len(arms))
    vals = [stats[a]["return_pct"][0] for a in arms]
    err = np.array([[stats[a]["return_pct"][0] - stats[a]["return_pct"][1] for a in arms],
                    [stats[a]["return_pct"][2] - stats[a]["return_pct"][0] for a in arms]])
    ax1.barh(y, vals, xerr=err, height=0.62, color=[C[a] for a in arms],
             error_kw={"ecolor": MUTED, "lw": 1.2})
    ax1.axvline(bh["return_pct"], color=MUTED, lw=1.4, ls="--")
    ax1.annotate(f' B&H {bh["return_pct"]:+.1f}%', (bh["return_pct"], len(arms) - 0.6),
                 color=MUTED, fontsize=8, fontweight="bold")
    ax1.set_yticks(y, arms)
    ax1.invert_yaxis()
    ax1.set_title("Test return IQM + 95% CI (10 seeds/arm)", color=INK, fontsize=10, loc="left")
    ax1.set_xlabel("test return %", color=MUTED, fontsize=8)

    for i, a in enumerate(arms):
        r, rlo, rhi = stats[a]["return_pct"]
        m, mlo, mhi = stats[a]["max_drawdown_pct"]
        ax2.errorbar(m, r, xerr=[[m - mlo], [mhi - m]], yerr=[[r - rlo], [rhi - r]],
                     fmt="o", color=C[a], ecolor="#c9c8bd", elinewidth=1, ms=7)
        dx, ha = (6, "left") if i % 2 == 0 else (-6, "right")
        ax2.annotate(a, (m, r), color=C[a], fontsize=8, fontweight="bold",
                     xytext=(dx, 5 + 3 * (i % 3)), textcoords="offset points", ha=ha)
    ax2.plot(bh["max_drawdown_pct"], bh["return_pct"], "s", color=MUTED, ms=8)
    ax2.annotate(" buy&hold", (bh["max_drawdown_pct"], bh["return_pct"]),
                 color=MUTED, fontsize=8, fontweight="bold")
    ax2.set_title("The risk plane: max drawdown vs return (test IQMs)",
                  color=INK, fontsize=10, loc="left")
    ax2.set_xlabel("max drawdown % (right = shallower)", color=MUTED, fontsize=8)
    ax2.set_ylabel("test return %", color=MUTED, fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "assets"), exist_ok=True)
    out = os.path.join(HERE, "assets", "hero.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def selfcheck():
    market = ne.Market()
    feat = torch.tensor(ne.build_features(market, OBS_ARM), device=DEV)

    # 1. substitution trick: net's greedy actions replayed through napkin-tape's
    #    independent CPU accounting must reproduce the vectorized equity curve
    import napkin_tape as nt
    torch.manual_seed(0)
    net = QNet(ne.obs_dim(OBS_ARM), 3).to(DEV)
    acts = torch.tensor(ACTION_SETS["long3"], device=DEV)
    t0, t1 = market.t_train_end, market.t_train_end + 40
    sym = "NVDA"; si = ne.UNIVERSE.index(sym)
    # extract single-symbol actions and curve from the vectorized eval
    pos = torch.zeros(1, device=DEV); eq = 1.0; gpu_curve = []; actions = []
    for t in range(t0, t1):
        o = ne.make_obs(feat, torch.tensor([t], device=DEV),
                        torch.tensor([si], device=DEV), pos, OBS_ARM)
        p = acts[net(o).argmax(1)]
        eq *= market.step_factor(torch.tensor([t], device=DEV),
                                 torch.tensor([si], device=DEV), pos, p).item()
        actions.append(p.item()); gpu_curve.append(eq); pos = p
    rows = [json.loads(l) for l in open(os.path.join(ne.TAPE, sym + ".bulk.jsonl"))]
    rows = [r for r in rows if r["date"] in set(market.dates)]
    tape1 = nt.Tape({sym: rows})
    k = {"i": 0}
    def pol(view):
        i = k["i"]; k["i"] += 1
        return {sym: actions[i]} if i < len(actions) else {}
    ref, _ = nt.run_sim(tape1, pol, warmup=t0)
    worst = max(abs(g - r / 100_000.0) / (r / 100_000.0)
                for g, r in zip(gpu_curve, ref[:len(gpu_curve)]))
    assert worst < 2e-3, f"substitution trick: {worst:.3%} deviation"
    print(f"selfcheck 1/4: substitution trick — net actions through napkin-tape's "
          f"accounting agree (worst {worst:.4%} over {len(actions)} bars)")

    # 2. reward identities on a scripted episode
    log_rs = torch.tensor([0.01, -0.03, 0.02, -0.01], device=DEV)
    eq_v = torch.cumprod(torch.exp(log_rs), 0)
    peak_v = torch.cummax(eq_v, 0).values
    assert torch.allclose(sum(shape_reward("pnl", r, None, None) for r in log_rs),
                          log_rs.sum())
    sh = sum(shape_reward("sharpe", r, None, None) for r in log_rs)
    assert torch.allclose(sh, log_rs.sum() - KAPPA_SH * (log_rs ** 2).sum())
    shape_reward.prev_dd = torch.zeros(1, device=DEV)
    pen = 0.0
    for i, r in enumerate(log_rs):
        rr = shape_reward("dd", r, eq_v[i:i+1], peak_v[i:i+1])
        shape_reward.prev_dd = 1.0 - eq_v[i:i+1] / peak_v[i:i+1]
        pen += (r - rr).item()
    # independent recomputation from the equity curve: penalty == lambda * total
    # POSITIVE variation of drawdown depth (not final depth — dd can recover)
    dd_series = (1.0 - eq_v / peak_v).cpu().numpy()
    expected = LAMBDA_DD * np.clip(np.diff(np.concatenate([[0.0], dd_series])), 0, None).sum()
    assert abs(pen - expected) < 1e-6, (pen, expected)
    print("selfcheck 2/4: reward identities (pnl == log-equity; sharpe penalty == "
          "kappa*sum(r^2); dd penalty == lambda*positive drawdown variation)")

    # 3. action-space honesty over full evals
    for aset, allowed in (("long2", {0.0, 1.0}), ("size5", {-1.0, -0.5, 0.0, 0.5, 1.0})):
        acts_v = torch.tensor(ACTION_SETS[aset], device=DEV)
        netv = QNet(ne.obs_dim(OBS_ARM), len(acts_v)).to(DEV)
        _, poss = evaluate(netv, acts_v, market, feat, market.t_val_end, market.T)
        vals = {v for row in poss for v in row}
        assert vals <= allowed, f"{aset} emitted {vals - allowed}"
    print("selfcheck 3/4: long2 never shorts; size5 only takes declared tiers")

    # 4. ratio bookkeeping on short runs
    for arm, ratio in (("ratio.25", 0.25), ("base", 1.0), ("ratio4", 4.0)):
        shape_reward.prev_dd = torch.zeros(N_ENVS, device=DEV)
        train(arm, 0, market, feat, vec_steps=200)
        added, updates = train.last_bookkeeping
        assert abs(updates * BATCH - added * ratio) <= BATCH, \
            f"{arm}: {updates} updates x {BATCH} vs {added} x {ratio}"
    print("selfcheck 4/4: gradient samples == ratio x transitions (±1 batch)")
    print("ALL SELFCHECKS PASS")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    if cmd == "sweep":
        sweep([a for a in sys.argv[2:] if a in ARMS] or None)
    else:
        {"selfcheck": selfcheck, "report": report, "plot": plot}[cmd]()
