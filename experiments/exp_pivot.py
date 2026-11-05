"""EXP-C: calibration of the exact Wishart pivot (Theorem 3).

For fixed queries with exact couplings, Delta/Delta_hat ~ chi^2_{d-r+1}/d.
We draw many independent sketches and check (i) coverage of the exact
(1-a) confidence intervals and (ii) unbiasedness of the debiased estimator.
"""
import sys, os, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from scipy.stats import chi2
from graphio import load
from core import Graph, SpanningTree, Solver, CycleSketch
from baselines import truth, coupling_exact, quad_pinv

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--d", type=int, nargs="+", default=[16, 64, 256])
ap.add_argument("--nq", type=int, default=60)
ap.add_argument("--reps", type=int, default=100)
args = ap.parse_args()

n, u, v, w = load(args.dataset)
G = Graph(n, u, v, w, args.dataset)
T = SpanningTree(G, kind="mst" if np.ptp(G.w) > 0 else "bfs")
S = Solver(G, T.root, "cholmod")
rng = np.random.default_rng(1)

# queries: k in {1,2,3}, F local around a random non-bridge edge, s near it
probe = CycleSketch(G, T, S, d=32, seed=999, keep_potentials=False)
mass = (probe.Y ** 2).sum(0)
nonbridge = np.flatnonzero(mass > 1e-12 * np.median(mass))
adj = G.adjacency().tocsr()
Q = []
while len(Q) < args.nq:
    k = 1 + len(Q) % 3
    e = int(rng.choice(nonbridge))
    nbr = np.flatnonzero(np.isin(G.u, [G.u[e], G.v[e]]) | np.isin(G.v, [G.u[e], G.v[e]]))
    nbr = nbr[nbr != e]
    if len(nbr) < k - 1:
        continue
    F = np.array([e] + list(rng.choice(nbr, k - 1, replace=False)))
    s = int(G.u[e]); t = int(rng.integers(G.n))
    b = np.zeros(G.n); b[s] = 1; b[t] = -1
    x, _ = S.solve(b)
    tr = truth(G, S, F, s, t, x=x)
    if math.isinf(tr["delta"]) or tr["delta"] <= 1e-12 * tr["R0"] or tr["rank"] == 0:
        continue
    Q.append({"F": F, "s": s, "t": t, "x": x, "delta": tr["delta"], "rank": tr["rank"], "mu": tr["mu"]})

res = {"dataset": args.dataset, "per_d": []}
for d in args.d:
    ratios = []   # Delta_hat / Delta
    ranks = []
    t0 = time.time()
    for rep in range(args.reps):
        sk = CycleSketch(G, T, S, d=d, seed=10_000 + rep, keep_potentials=False)
        for q in Q:
            qq = sk.query(q["F"], q["s"], q["t"], x=q["x"], mode="exact")
            ratios.append(qq["delta"] / q["delta"]); ranks.append(qq["rank"])
    ratios = np.array(ratios); ranks = np.array(ranks)
    cover = {}
    for a in [0.10, 0.05, 0.01]:
        lo = chi2.ppf(a / 2, d - ranks + 1) / d
        hi = chi2.ppf(1 - a / 2, d - ranks + 1) / d
        # Delta in [Dhat*lo, Dhat*hi]  <=> 1/ratio in [lo, hi]
        inv = 1 / ratios
        cover[str(1 - a)] = float(np.mean((inv >= lo) & (inv <= hi)))
    deb = ratios * (d - ranks - 1) / d
    rec = {"d": d, "n_estimates": int(len(ratios)), "coverage": cover,
           "mean_ratio_raw": float(ratios.mean()), "mean_ratio_debiased": float(deb.mean()),
           "se_mean_debiased": float(deb.std() / math.sqrt(len(deb))),
           "rel_std_debiased": float(deb.std()), "predicted_rel_std": float(np.mean(np.sqrt(2 / np.maximum(d - ranks - 3, 1)))),
           "time_s": time.time() - t0}
    res["per_d"].append(rec)
    print(json.dumps(rec), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", f"pivot_{args.dataset}.json"), "w"), indent=1)
