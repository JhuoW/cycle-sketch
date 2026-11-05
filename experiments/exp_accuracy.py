"""EXP-A: accuracy of what-if resistance responses versus failure margin.

For each dataset we build ONE set of d Gaussian probes and compare estimators
that use exactly the same solves:
  ours-exact : cycle sketch (lifted) + exact pair coupling   [CycleSketch]
  ours-tree  : cycle sketch + tree-routing correction (no per-pair solve)
  cut        : same probes, keep the cut part, C_FF = I - P_FF  [JL/SS-type]
  raw-cycle  : cycle part WITHOUT lift, inexact solver (tol)    [ablation]
  fixedpoly  : original ERGraph3 fixed polynomial (small graphs only)
  wood-iter  : online exact Woodbury with an iterative solver at tol 1e-6
Ground truth: CHOLMOD Woodbury with exact topology (refactorization when the
supported failure matrix is ill-conditioned).

Workloads (per k in {1,2,3}):
  'local'   : F = an edge e plus k-1 edges within 2 hops of e; s within 2 hops
              of e's endpoint, t uniform at random.
  'random'  : s,t uniform, F uniform.
Edges e are drawn half uniformly and half from near-bridge candidates (lowest
estimated cycle mass from an INDEPENDENT pilot sketch, different seed);
evaluation strata use the exact failure margin mu, so the selection does not
bias the measured errors.
"""
import sys, os, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from graphio import load
from core import Graph, SpanningTree, Solver, CycleSketch
from baselines import CutGram, RawCycleGram, quad_pinv, coupling_exact, truth, FixedPolySolver, woodbury_CFF

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--d", type=int, nargs="+", default=[64, 256])
ap.add_argument("--nq", type=int, default=150, help="queries per (workload,k)")
ap.add_argument("--solver", default="cholmod")
ap.add_argument("--iter_solver", default="amg")
ap.add_argument("--raw_tol", type=float, default=1e-4)
ap.add_argument("--fixedpoly", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip_online", action="store_true")
ap.add_argument("--raw_d", type=int, nargs="*", default=None)
ap.add_argument("--out", default=None, help="output JSON (default results/accuracy_<dataset>.json)")
args = ap.parse_args()
# Large weighted grids: the online iterative baselines (AMG, ~1.7 s/solve) were already
# measured on PEG9241/ACT25k; skip them here to keep the run tractable.
if args.dataset in os.environ.get("SKIP_ONLINE_FOR", "ACT70k").split(","):
    args.skip_online = True
    if args.raw_d is None:
        args.raw_d = [64]

rng = np.random.default_rng(args.seed)
n, u, v, w = load(args.dataset)
G = Graph(n, u, v, w, args.dataset)
weighted = bool(np.ptp(G.w) > 0)
T = SpanningTree(G, kind="mst" if weighted else "bfs")
S = Solver(G, T.root, args.solver)
Sit = Solver(G, T.root, args.iter_solver)
print(f"[{args.dataset}] n={G.n} m={G.m} solver={args.solver} setup={S.setup_time:.2f}s", flush=True)

# --- independent pilot sketch to find near-bridge candidates ---------------
pilot = CycleSketch(G, T, S, d=32, seed=10_000 + args.seed, keep_potentials=False)
cmass = (pilot.Y ** 2).sum(0)
nonbridge = np.flatnonzero(cmass > 1e-12 * np.median(cmass[cmass > 0]))   # bridges have C_ee = 0
nb_cand = nonbridge[np.argsort(cmass[nonbridge])[: max(200, G.m // 50)]]
del pilot

adj = G.adjacency().tocsr()


def ball(x, r=2):
    seen = {int(x)}; frontier = [int(x)]
    for _ in range(r):
        nxt = []
        for a in frontier:
            for b in adj.indices[adj.indptr[a]:adj.indptr[a + 1]]:
                if b not in seen:
                    seen.add(int(b)); nxt.append(int(b))
        frontier = nxt
    return np.array(sorted(seen))


def incident_edges(vs):
    vs = set(int(a) for a in vs)
    return np.flatnonzero(np.isin(G.u, list(vs)) | np.isin(G.v, list(vs)))


queries = []
WLS = os.environ.get("WORKLOADS", "local,random").split(",")
for k in [1, 2, 3]:
    for wl in WLS:
        for qi in range(args.nq):
            if wl == "endpoint":
                e = int(rng.choice(nb_cand)) if qi % 2 == 0 else int(rng.integers(G.m))
                s = int(G.u[e]) if rng.random() < 0.5 else int(G.v[e])
                near = incident_edges([G.u[e], G.v[e]])
                near = near[near != e]
                F = [e] + (list(rng.choice(near, k - 1, replace=False)) if k > 1 and len(near) >= k - 1 else list(rng.choice(G.m, k - 1, replace=False)))
                t = int(rng.integers(G.n))
            elif wl == "local":
                e = int(rng.choice(nb_cand)) if qi % 2 == 0 else int(rng.integers(G.m))
                B = ball(G.u[e], 2)
                near = incident_edges(B)
                near = near[near != e]
                F = [e] + (list(rng.choice(near, k - 1, replace=False)) if k > 1 and len(near) >= k - 1 else list(rng.choice(G.m, k - 1, replace=False)))
                s = int(rng.choice(B)); t = int(rng.integers(G.n))
            else:
                F = list(rng.choice(G.m, k, replace=False))
                s, t = (int(a) for a in rng.choice(G.n, 2, replace=False))
            if s == t:
                t = (t + 1) % G.n
            queries.append({"k": k, "wl": wl, "F": [int(a) for a in F], "s": s, "t": t})

# --- ground truth ------------------------------------------------------------
t0 = time.time()
xcache = {}
for q in queries:
    b = np.zeros(G.n); b[q["s"]] = 1; b[q["t"]] = -1
    x, _ = S.solve(b)
    q["R0"] = float(x[q["s"]] - x[q["t"]])
    tr = truth(G, S, q["F"], q["s"], q["t"], x=x, R0=q["R0"])
    q["delta"], q["mu"], q["rank"] = tr["delta"], tr["mu"], tr["rank"]
    q["fF"] = coupling_exact(G, x, np.array(q["F"]))      # O(k) storage instead of an n-vector
print(f"  truth for {len(queries)} queries in {time.time()-t0:.1f}s", flush=True)

records = []


def rec(q, method, d, est, sep_pred=None, extra=None):
    r = {"dataset": args.dataset, "method": method, "d": d, "k": q["k"], "wl": q["wl"],
         "R0": q["R0"], "delta": q["delta"], "mu": q["mu"], "est": est}
    if sep_pred is not None:
        r["sep_pred"] = bool(sep_pred)
    if extra:
        r.update(extra)
    records.append(r)


# --- level estimator: re-solve on G' with Jacobi-PCG (tol 1e-4) and subtract ---
from core import block_pcg
import scipy.sparse as sp_
def pcg_resistance(Lmat, s, t, tol, keep_mask=None):
    n_ = Lmat.shape[0]
    idx = np.flatnonzero(np.arange(n_) != t)
    A_ = Lmat[idx][:, idx].tocsr()
    dg = A_.diagonal(); dg[dg <= 0] = 1.0
    rhs = np.zeros((len(idx), 1)); rhs[np.searchsorted(idx, s), 0] = 1.0
    X_, its_ = block_pcg(A_, rhs, 1.0 / dg, tol=tol, maxiter=20000)
    return float(X_[np.searchsorted(idx, s), 0]), its_[0]
t0 = time.time()
for qi, q in enumerate(queries):
    if args.skip_online:
        break
    if math.isinf(q["delta"]) or qi % 3 != 0:
        continue
    F = np.array(q["F"])
    w2 = G.w.copy(); w2[F] = 0.0
    G2 = Graph(G.n, G.u[w2 > 0], G.v[w2 > 0], w2[w2 > 0])
    R1, _ = pcg_resistance(G2.L, q["s"], q["t"], 1e-4)
    R0h, _ = pcg_resistance(G.L, q["s"], q["t"], 1e-4)
    rec(q, "resolve-pcg(1e-4)", 0, R1 - R0h)
print(f"  resolve-pcg done in {time.time()-t0:.1f}s", flush=True)

# --- online exact Woodbury with an iterative solver -------------------------
t0 = time.time()
for q in queries:
    if args.skip_online:
        break
    if math.isinf(q["delta"]):
        continue
    F = np.array(q["F"])
    CFF, its = woodbury_CFF(G, Sit, F, tol=1e-6)
    est = quad_pinv(CFF, q["fF"], q["rank"])
    rec(q, "wood-iter(1e-6)", 0, est)
t_wood = (time.time() - t0) / max(1, len(queries))

# Restrict stored columns to the edges used by the workload (estimates are unchanged:
# all methods share the probes, and columns are independent of the catalog).
CAT = np.unique(np.concatenate([np.array(q["F"], dtype=np.int64) for q in queries]))
for d in args.d:
    t0 = time.time()
    sk = CycleSketch(G, T, S, d=d, seed=args.seed + 1, catalog=CAT, store_dtype=np.float32)
    tb = time.time() - t0
    cut = CutGram(G, S, d=d, seed=args.seed + 1, catalog=CAT)
    raw = RawCycleGram(G, Sit, d=d, seed=args.seed + 1, tol=args.raw_tol, catalog=CAT) if (args.raw_d is None or d in args.raw_d) else None
    fp = None
    if args.fixedpoly:
        fps = FixedPolySolver(G, T, beta_star=1e-2)
        t1 = time.time(); fp = CycleSketch(G, T, fps, d=d, seed=args.seed + 1, keep_potentials=False, catalog=CAT)
        print(f"  fixedpoly degree={fps.degree} build={time.time()-t1:.1f}s", flush=True)
    print(f"  d={d}: sketch build {tb:.2f}s xi={sk.xi:.2e}", flush=True)
    for q in queries:
        F = np.array(q["F"]); s, t = q["s"], q["t"]
        qe = sk.query(F, s, t, fF=q["fF"])
        rec(q, "ours-exact", d, qe["delta"], qe["separated"])
        qd = sk.query(F, s, t, fF=q["fF"], debias=True)
        rec(q, "ours-exact-debiased", d, qd["delta"], qd["separated"])
        qt = sk.query(F, s, t, mode="tree")
        rec(q, "ours-tree", d, qt["delta"], qt["separated"], {"excess_bound": qt.get("excess_bound"), "RT": qt.get("excess_bound")})
        if math.isinf(q["delta"]):
            continue
        z = q["fF"]
        rec(q, "cut", d, quad_pinv(cut.CFF(F), z, q["rank"]))
        if raw is not None:
            rec(q, f"raw-cycle({args.raw_tol:g})", d, quad_pinv(raw.CFF(F), z, q["rank"]))
        if fp is not None:
            YF = fp.Y[:, fp.pos[F]]
            rec(q, "fixedpoly(b=1e-2)", d, quad_pinv(YF.T @ YF, z, q["rank"]))

out = args.out or os.path.join(os.path.dirname(__file__), "..", "results", f"accuracy_{args.dataset}.json")
json.dump({"dataset": args.dataset, "n": G.n, "m": G.m, "records": records, "t_wood_iter_per_query": t_wood},
          open(out, "w"), default=float)

# quick summary
import collections
agg = collections.defaultdict(list)
for r in records:
    if math.isinf(r["delta"]):
        agg[(r["method"], r["d"], "sep")].append(1.0 if r.get("sep_pred") else 0.0)
        continue
    floor = max(r["delta"], 1e-3 * r["R0"])
    err = abs(r["est"] - r["delta"]) / floor if np.isfinite(r["est"]) else np.inf
    stratum = "mu<1e-2" if r["mu"] < 1e-2 else "mu>=1e-2"
    agg[(r["method"], r["d"], stratum)].append(err)
for key in sorted(agg, key=str):
    a = np.array(agg[key])
    if key[2] == "sep":
        print(f"  {key}: separated detected {a.mean():.3f} (n={len(a)})")
    else:
        print(f"  {key}: median {np.median(a):.3g}  p90 {np.quantile(a,0.9):.3g}  max {a.max():.3g}  (n={len(a)})")
