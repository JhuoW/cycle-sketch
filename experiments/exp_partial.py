"""EXP-F: partial failures (conductance reduced by a fraction alpha).

As alpha -> 1 a weakened near-bridge approaches disconnection; the failure
matrix I - D + D^{1/2} C_FF D^{1/2} approaches the cycle Gram matrix.  We compare
CycleSketch and CutSketch (same probes) against exact truth for
alpha in {0.5, 0.9, 0.99, 0.999, 1}.
"""
import sys, os, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from graphio import load
from core import Graph, SpanningTree, Solver, CycleSketch
from baselines import CutGram, truth, coupling_exact, quad_pinv

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--d", type=int, default=256)
ap.add_argument("--nq", type=int, default=150)
args = ap.parse_args()
n, u, v, w = load(args.dataset)
G = Graph(n, u, v, w, args.dataset)
T = SpanningTree(G, kind="mst" if np.ptp(G.w) > 0 else "bfs")
S = Solver(G, T.root, "cholmod")
pilot = CycleSketch(G, T, S, d=32, seed=777, keep_potentials=False, spectral_certificate=False)
mass = (pilot.Y ** 2).sum(0)
del pilot
nonbridge = np.flatnonzero(mass > 1e-12 * np.median(mass))
low = nonbridge[np.argsort(mass[nonbridge])[: max(100, G.m // 50)]]
rng = np.random.default_rng(4)
QS = []
for qi in range(args.nq):
    k = 1 + qi % 2
    e = int(rng.choice(low)) if qi % 3 else int(rng.choice(nonbridge))
    nbr = np.flatnonzero(np.isin(G.u, [G.u[e], G.v[e]]) | np.isin(G.v, [G.u[e], G.v[e]]))
    nbr = nbr[nbr != e]
    F = np.array([e] + (list(rng.choice(nbr, 1)) if k == 2 and len(nbr) else []))
    QS.append((F, int(G.u[e]), int(rng.integers(G.n))))
CAT = np.unique(np.concatenate([q[0] for q in QS]))
sk = CycleSketch(G, T, S, d=args.d, seed=21, catalog=CAT, keep_potentials=False)
cut = CutGram(G, S, d=args.d, seed=21, catalog=CAT)
recs = []
for F, s, t in QS:
    b = np.zeros(G.n); b[s] = 1; b[t] = -1
    x, _ = S.solve(b)
    for a in [0.5, 0.9, 0.99, 0.999, 1.0]:
        alpha = np.full(len(F), a)
        tr = truth(G, S, F, s, t, alpha=alpha, x=x)
        if math.isinf(tr["delta"]) or tr["delta"] <= 0:
            continue
        q = sk.query(F, s, t, alpha=alpha, x=x, mode="exact")
        D = np.sqrt(alpha)
        Mc = np.eye(len(F)) - np.diag(alpha) + D[:, None] * cut.CFF(F) * D[None, :]
        z = D * coupling_exact(G, x, F)
        ec = quad_pinv(Mc, z, tr["rank"])
        recs.append({"alpha": a, "k": int(len(F)), "delta": tr["delta"], "R0": tr["R0"], "mu": tr["mu"],
                     "ours": q["delta"], "cut": ec})
json.dump({"dataset": args.dataset, "d": args.d, "records": recs},
          open(os.path.join(os.path.dirname(__file__), "..", "results", f"partial_{args.dataset}.json"), "w"), default=float)
for a in [0.5, 0.9, 0.99, 0.999, 1.0]:
    rr = [r for r in recs if r["alpha"] == a and r["delta"] >= 1e-3 * r["R0"]]
    eo = [abs(r["ours"] - r["delta"]) / r["delta"] for r in rr]
    ec = [abs(r["cut"] - r["delta"]) / r["delta"] if np.isfinite(r["cut"]) and r["cut"] >= 0 else np.inf for r in rr]
    mu = [r["mu"] for r in rr]
    print(f"alpha={a}: n={len(rr)} median mu={np.median(mu):.2e}  ours med {np.median(eo):.3f} p90 {np.quantile(eo,.9):.3f} | "
          f"cut med {np.median(ec):.3f} p90 {np.quantile(ec,.9):.3f} fail {np.mean(~np.isfinite(ec)):.2f}")
