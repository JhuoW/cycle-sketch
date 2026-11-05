"""EXP-D: sketch-corrected couplings (Theorem 4) versus the uncorrected
approximate-potential coupling used by the original ERGraph3 interval.

Pair potentials come from j iterations of an iterative solver (j small), i.e.
cheap, inexact pair preparation.  We compare
  uncorrected : z~ = Phi x~ on F, Delta_hat = z~^T C_hat^+ z~   (+ ERGraph3 interval)
  corrected   : f^ = p_F - Y_F^T Pi r,  p = Phi x~ + T r         (+ our interval)
  tree        : x~ = 0 (index-only, no solve)
Both intervals use the same per-query embedding parameter zeta.
"""
import sys, os, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from graphio import load
from core import Graph, SpanningTree, Solver, CycleSketch, residual_full
from baselines import truth, quad_pinv

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--d", type=int, default=256)
ap.add_argument("--nq", type=int, default=120)
ap.add_argument("--iters", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
ap.add_argument("--iter", default="amg")
args = ap.parse_args()

n, u, v, w = load(args.dataset)
G = Graph(n, u, v, w, args.dataset)
T = SpanningTree(G, kind="mst" if np.ptp(G.w) > 0 else "bfs")
S = Solver(G, T.root, "cholmod")
Sit = Solver(G, T.root, args.iter)
pilot = CycleSketch(G, T, S, d=32, seed=303, keep_potentials=False, spectral_certificate=False)
rng = np.random.default_rng(2)
mass = (pilot.Y ** 2).sum(0)
del pilot
nonbridge = np.flatnonzero(mass > 1e-12 * np.median(mass))
low = nonbridge[np.argsort(mass[nonbridge])[: max(100, G.m // 50)]]

Q = []
while len(Q) < args.nq:
    k = 1 + len(Q) % 3
    e = int(rng.choice(low)) if len(Q) % 2 == 0 else int(rng.choice(nonbridge))
    nbr = np.flatnonzero(np.isin(G.u, [G.u[e], G.v[e]]) | np.isin(G.v, [G.u[e], G.v[e]]))
    nbr = nbr[nbr != e]
    if len(nbr) < k - 1:
        continue
    F = np.array([e] + list(rng.choice(nbr, k - 1, replace=False)))
    s = int(G.u[e]); t = int(rng.integers(G.n))
    b = np.zeros(G.n); b[s] = 1; b[t] = -1
    x, _ = S.solve(b)
    tr = truth(G, S, F, s, t, x=x)
    if math.isinf(tr["delta"]) or tr["rank"] == 0:
        continue
    Q.append({"F": F, "s": s, "t": t, "b": b, "delta": tr["delta"], "R0": tr["R0"], "mu": tr["mu"], "k": k})


CAT = np.unique(np.concatenate([q["F"] for q in Q]))
sk = CycleSketch(G, T, S, d=args.d, seed=3, catalog=CAT, store_dtype=np.float32)


def zeta_q(d, k, delta=0.05):
    return (math.sqrt(k + 1) + math.sqrt(2 * math.log(2 / delta))) / math.sqrt(d)


records = []
for q in Q:
    F, s, t, b = q["F"], q["s"], q["t"], q["b"]
    zeta = zeta_q(args.d, len(F)); zp = zeta + sk.xi; kap = (1 + zp) ** 2 - 1
    # tree mode
    qt = sk.query(F, s, t, mode="tree")
    excess = math.sqrt(max(T.lca_resistance(s, t), 0.0))
    lo = max(0.0, (1 - zp) * math.sqrt(qt["delta"]) - kap * excess) ** 2
    hi = ((1 + zp) * math.sqrt(qt["delta"]) + kap * excess) ** 2
    records.append({**{k_: q[k_] for k_ in ["delta", "R0", "mu", "k"]}, "mode": "tree", "j": 0, "eta": excess,
                    "est": qt["delta"], "lo": lo, "hi": hi})
    for j in args.iters:
        bg = b.copy()
        if Sit.kind == "amg":
            xg = np.zeros(G.n)
            xg[Sit.keep] = Sit.ml.solve(b[Sit.keep], tol=1e-300, accel="cg", maxiter=j)
        else:
            xg, _ = Sit.solve(b, tol=1e-300, maxiter=max(1, j // 20))
        r = residual_full(G, xg, b)
        _, e2 = T.route(r[:, None]); eta = math.sqrt(float(e2[0]))
        # corrected
        qc = sk.query(F, s, t, x=xg, r=r, mode="approx")
        lo = max(0.0, (1 - zp) * math.sqrt(qc["delta"]) - kap * eta) ** 2
        hi = ((1 + zp) * math.sqrt(qc["delta"]) + kap * eta) ** 2
        records.append({**{k_: q[k_] for k_ in ["delta", "R0", "mu", "k"]}, "mode": "corrected", "j": j, "eta": eta,
                        "est": qc["delta"], "lo": lo, "hi": hi})
        # uncorrected (ERGraph3 Sec. 5.9 interval)
        qu = sk.query(F, s, t, x=xg, mode="exact")
        h = math.sqrt(qu["delta"]); mu_hat = qu.get("smin2", np.nan)
        am, ap_ = (1 - zp) ** 2, (1 + zp) ** 2
        gam = eta * math.sqrt(ap_ / mu_hat) if mu_hat > 0 else math.inf
        lo = max(0.0, math.sqrt(am) * h - gam) ** 2
        hi = (math.sqrt(ap_) * h + gam) ** 2
        records.append({**{k_: q[k_] for k_ in ["delta", "R0", "mu", "k"]}, "mode": "uncorrected", "j": j, "eta": eta,
                        "est": qu["delta"], "lo": lo, "hi": hi})

json.dump({"dataset": args.dataset, "d": args.d, "records": records},
          open(os.path.join(os.path.dirname(__file__), "..", "results", f"coupling_{args.dataset}.json"), "w"), default=float)
import collections
agg = collections.defaultdict(list)
for r in records:
    err = abs(r["est"] - r["delta"]) / max(r["delta"], 1e-3 * r["R0"])
    width = (r["hi"] - r["lo"]) / max(r["delta"], 1e-3 * r["R0"])
    cov = r["lo"] <= r["delta"] <= r["hi"]
    agg[(r["mode"], r["j"], "mu<1e-2" if r["mu"] < 1e-2 else "mu>=1e-2")].append((err, width, cov, r["eta"]))
for key in sorted(agg, key=str):
    a = np.array(agg[key])
    print(f"{key}: err med {np.median(a[:,0]):.3g} p90 {np.quantile(a[:,0],.9):.3g} | width med {np.median(a[:,1]):.3g} | cover {a[:,2].mean():.2f} | eta med {np.median(a[:,3]):.2g} n={len(a)}")
