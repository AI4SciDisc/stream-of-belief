"""Animate a belief trajectory (from ``sob demo --trace``) as a GIF: beliefs flowing on a PCA plane."""
from __future__ import annotations

import json

import numpy as np

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _ease(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)


def make_gif(trace_path: str, out: str, fps: float = 12.0, steps: int = 12, hold: int = 10,
             title: str | None = None):
    """``steps`` interpolated frames per cycle, ``hold`` frames of pause at every cycle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    tr = json.load(open(trace_path))
    cycles = tr["result"]["cycles"]
    prob = tr["problem"]
    gold = prob.get("answer")
    P = np.array([c["pooled"] for c in cycles], dtype=np.float64)   # (K+1, N, d)
    K1, N, d = P.shape
    X = P.reshape(-1, d)
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Y = (X @ Vt[:2].T).reshape(K1, N, 2)
    res = [c["residual"] for c in cycles]

    # frame schedule: (cycle index k, interpolation 0..1 between k-1 and k)
    sched = [(0, 1.0)] * hold
    for k in range(1, K1):
        sched += [(k, s) for s in _ease(np.linspace(0, 1, steps, endpoint=False))] + [(k, 1.0)] * hold
    sched += [(K1 - 1, 1.0)] * hold

    fig, (ax, ar) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw=dict(width_ratios=[1.45, 1]),
                                 constrained_layout=True)
    fig.suptitle(title or f"Stream of Belief  ·  {N} agents  ·  {prob.get('family', '')}", fontweight="bold")
    span = Y.max((0, 1)) - Y.min((0, 1)) + 1e-6
    ax.set_xlim(Y[..., 0].min() - 0.15 * span[0], Y[..., 0].max() + 0.25 * span[0])
    ax.set_ylim(Y[..., 1].min() - 0.15 * span[1], Y[..., 1].max() + 0.2 * span[1])
    ax.set_xlabel("belief space, PC 1"); ax.set_ylabel("belief space, PC 2")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False); ar.spines[s].set_visible(False)
    ar.set_xlim(0.5, max(1.5, K1 - 0.5)); ar.set_yscale("log")
    rv = [r for r in res[1:] if r and r > 0]
    ar.set_ylim(min(rv) / 2 if rv else 1e-3, max(rv) * 2 if rv else 1)
    ar.set_xlabel("cycle $k$"); ar.set_ylabel("belief residual $r_k$")
    ar.set_xticks(range(1, K1))
    trails = [ax.plot([], [], "-", color=COLORS[i % 8], alpha=0.45, lw=1.6)[0] for i in range(N)]
    ghosts = [ax.plot([], [], "o", color=COLORS[i % 8], alpha=0.25, ms=5)[0] for i in range(N)]
    dots = [ax.plot([], [], "o", color=COLORS[i % 8], ms=15, mec="white", mew=1.2, zorder=5 + i)[0] for i in range(N)]
    letters = [ax.text(0, 0, chr(65 + i), color="white", fontsize=8, fontweight="bold", ha="center", va="center",
                       zorder=10 + i) for i in range(N)]
    legend = [ax.text(0.985, 0.97 - 0.065 * i, "", color=COLORS[i % 8], fontsize=10, fontweight="bold", ha="right",
                      va="top", transform=ax.transAxes) for i in range(N)]
    rline, = ar.plot([], [], "o-", color="#1f77b4", lw=1.8)
    status = ax.set_title("", loc="left", fontsize=10)

    def frame(f):
        k, s = sched[f]
        cur = Y[k] if k == 0 else (1 - s) * Y[k - 1] + s * Y[k]
        for i in range(N):
            xs = list(Y[:k, i, 0]) + [cur[i, 0]]; ys = list(Y[:k, i, 1]) + [cur[i, 1]]
            trails[i].set_data(xs, ys)
            ghosts[i].set_data(Y[:k, i, 0], Y[:k, i, 1])
            dots[i].set_data([cur[i, 0]], [cur[i, 1]])
            kk = k if s >= 1.0 else k - 1
            a = cycles[kk]["probe_answers"][i] or "·"
            letters[i].set_position((cur[i, 0], cur[i, 1]))
            legend[i].set_text(f"{chr(65 + i)}: {a}")
        kk = k if s >= 1.0 else k - 1
        rline.set_data(range(1, kk + 1), res[1:kk + 1])
        maj = cycles[kk]["probe_majority"]
        ok = "" if gold is None else ("  ✓" if str(maj).lower() == str(gold).lower() else "  ✗")
        status.set_text(f"cycle {kk}   ·   collective answer: {maj}{ok}")
        return trails + ghosts + dots + letters + legend + [rline, status]

    anim = FuncAnimation(fig, frame, frames=len(sched), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
