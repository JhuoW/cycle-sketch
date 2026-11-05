"""EXP-B: solver-agnostic certificate.

Same Gaussian probes, different solvers.  For each solver configuration we
build the lifted sketch (ours) and the unlifted cycle sample ('raw'), and
compare each against the exact-solve sketch built from the SAME probes, so the
measured deviation is purely solver-induced.  Theorem 2 predicts
  | ||Y_cfg c|| - ||Y_exact c|| | <= xi * ||C e||      (lifted, any solver)
and no such relative bound for the raw sample.
"""
import sys, os, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from graphio import load
from core import Graph, SpanningTree, Solver, CycleSketch
from baselines import RawCycleGram, woodbury_CFF, FixedPolySolver, edge_cols

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--iter", default="amg")
ap.add_argument("--tols", type=float, nargs="+", default=[1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
ap.add_argument("--nedges", type=int, default=3000)
ap.add_argument("--fixedpoly", action="store_true")
args = ap.parse_args()

n, u, v, w = load(args.dataset)
G = Graph(n, u, v, w, args.dataset)
weighted = bool(np.ptp(G.w) > 0)
T = SpanningTree(G, kind="mst" if weighted else "bfs")
S = Solver(G, T.root, "cholmod")
Sit = Solver(G, T.root, args.iter)
rng = np.random.default_rng(0)

# exact-probe reference sketch on all edges
ref = CycleSketch(G, T, S, d=args.d, seed=5, keep_potentials=False)
mass = (ref.Y ** 2).sum(0)
nonbridge = np.flatnonzero(mass > 1e-12 * np.median(mass))
low = nonbridge[np.argsort(mass[nonbridge])[: args.nedges // 2]]
uni = rng.choice(nonbridge, args.nedges // 2, replace=False)
E = np.unique(np.concatenate([low, uni]))
# exact C_ee for the evaluated edges
Cee = np.empty(len(E))
for b0 in range(0, len(E), 256):
    blk = E[b0:b0 + 256]
    X, _ = edge_cols(G, S, blk)
    Pee = G.sw[blk] * (X[G.u[blk], np.arange(len(blk))] - X[G.v[blk], np.arange(len(blk))])
    Cee[b0:b0 + 256] = 1 - Pee
refn = np.linalg.norm(ref.Y[:, E], axis=0)

configs = [(args.iter, t) for t in args.tols]
out = {"dataset": args.dataset, "d": args.d, "n": G.n, "m": G.m, "configs": []}
for (kind, tol) in configs + ([("fixedpoly", b) for b in [1e-1, 1e-2]] if args.fixedpoly else []):
    if kind == "fixedpoly":
        solver = FixedPolySolver(G, T, beta_star=tol)
    else:
        solver = Sit
    t0 = time.time()
    sk = CycleSketch(G, T, solver, d=args.d, seed=5, tol=tol, keep_potentials=False)
    tb = time.time() - t0
    raw = RawCycleGram(G, solver, d=args.d, seed=5, tol=tol)
    dev_l = np.abs(np.linalg.norm(sk.Y[:, E], axis=0) - refn) / np.sqrt(Cee)
    dev_r = np.abs(np.linalg.norm(raw.Y[:, E], axis=0) - refn) / np.sqrt(Cee)
    # error of the diagonal estimate relative to the exact C_ee
    rel_l = np.abs((sk.Y[:, E] ** 2).sum(0) / Cee - 1)
    rel_r = np.abs((raw.Y[:, E] ** 2).sum(0) / Cee - 1)
    rel_ref = np.abs((ref.Y[:, E] ** 2).sum(0) / Cee - 1)
    lowmask = Cee < np.quantile(Cee, 0.1)
    rec = {"solver": kind, "tol": tol, "build_s": tb, "mean_iters": float(np.mean(sk.iters)),
           "xi": sk.xi, "xi_frob": sk.xi_frob, "dev_lift_max": float(dev_l.max()), "dev_lift_med": float(np.median(dev_l)),
           "dev_raw_max": float(dev_r.max()), "dev_raw_med": float(np.median(dev_r)),
           "certificate_holds": bool(np.all(dev_l <= sk.xi * (1 + 1e-6) + 1e-12)),
           "relerr_diag_lift_med": float(np.median(rel_l)), "relerr_diag_lift_lowCee_med": float(np.median(rel_l[lowmask])),
           "relerr_diag_raw_med": float(np.median(rel_r)), "relerr_diag_raw_lowCee_med": float(np.median(rel_r[lowmask])),
           "relerr_diag_ref_med": float(np.median(rel_ref)), "relerr_diag_ref_lowCee_med": float(np.median(rel_ref[lowmask])),
           "Cee_min": float(Cee.min()), "Cee_q10": float(np.quantile(Cee, 0.1))}
    out["configs"].append(rec)
    print(json.dumps(rec), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"certificate_{args.dataset}.json"), "w"), indent=1)
