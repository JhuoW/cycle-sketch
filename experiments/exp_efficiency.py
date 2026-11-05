"""EXP-E: efficiency and scalability.

Measures (single-threaded numpy/scipy unless the BLAS uses threads):
  * index construction: tree, labels, solver setup, d solves, lift; bytes
  * query latency: exact-coupling mode (cached pair potentials), tree mode
  * vectorized all-edge single-failure throughput
  * baselines: online Woodbury (k solves/query), re-solve on G\\F (fresh
    factorization), LODF-cache build (= one solve per catalog edge), and the
    original fixed-polynomial construction (degree from the trace bound).
Large-graph baseline costs that would take days are EXTRAPOLATED from measured
per-unit costs and are labelled as such in the output.
"""
import sys, os, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from graphio import load
from core import Graph, SpanningTree, Solver, CycleSketch
from baselines import (woodbury_CFF, coupling_exact, quad_pinv, resistance_on_subgraph, FixedPolySolver,
                       apply_fixed_polynomial, total_stretch, fixed_poly_degree, edge_cols)

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--solver", default="cholmod")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--nq", type=int, default=2000)
ap.add_argument("--resolve_q", type=int, default=3)
ap.add_argument("--wood_q", type=int, default=30)
ap.add_argument("--skip_baselines", action="store_true", help="measure only the index (build, size, queries, scan)")
ap.add_argument("--out", default=None, help="output JSON (default results/efficiency_<ds>_d<d>.json)")
args = ap.parse_args()
res = {"dataset": args.dataset, "d": args.d, "solver": args.solver}

t0 = time.time(); n, u, v, w = load(args.dataset); G = Graph(n, u, v, w, args.dataset); res["t_load"] = time.time() - t0
res["n"], res["m"] = G.n, G.m
weighted = bool(np.ptp(G.w) > 0)
t0 = time.time(); T = SpanningTree(G, kind="mst" if weighted else "bfs"); res["t_tree"] = time.time() - t0
t0 = time.time(); S = Solver(G, T.root, args.solver); res["t_solver_setup"] = time.time() - t0
print(f"[{args.dataset}] n={G.n} m={G.m} tree {res['t_tree']:.1f}s solver-setup {res['t_solver_setup']:.1f}s", flush=True)
dt = np.float32 if args.dtype == "float32" else np.float64
t0 = time.time()
sk = CycleSketch(G, T, S, d=args.d, seed=1, tol=1e-10, store_dtype=dt, verbose=False)
res["t_build_total"] = time.time() - t0 + res["t_tree"] + res["t_solver_setup"]
res["t_build_solves"], res["t_build_lift"] = sk.t_solve, sk.t_lift
res["index_bytes"] = sk.nbytes(); res["xi"] = sk.xi
res["mean_iters"] = float(np.mean(sk.iters))
print(f"  build: total {res['t_build_total']:.1f}s solves {sk.t_solve:.1f}s lift {sk.t_lift:.1f}s "
      f"index {sk.nbytes()/1e9:.2f}GB xi={sk.xi:.1e}", flush=True)

rng = np.random.default_rng(7)
# monitored pairs and their potentials
P = 8
pairs = [tuple(int(a) for a in rng.choice(G.n, 2, replace=False)) for _ in range(P)]
t0 = time.time()
B = np.zeros((G.n, P))
for j, (s, t) in enumerate(pairs):
    B[s, j] = 1; B[t, j] = -1
X, _ = S.solve(B, tol=1e-10)
res["t_pair_prep_per_pair"] = (time.time() - t0) / P
# random queries: F = k edges incident to a random vertex neighbourhood of a random edge on the pair's tree path area
queries = []
for qi in range(args.nq):
    k = 1 + qi % 3
    j = qi % P
    e = int(rng.integers(G.m))
    nbr = np.flatnonzero((G.u == G.u[e]) | (G.v == G.u[e]) | (G.u == G.v[e]) | (G.v == G.v[e])) if qi < 200 else None
    if nbr is not None and len(nbr) >= k:
        F = rng.choice(nbr, k, replace=False)
    else:
        F = rng.choice(G.m, k, replace=False)
    queries.append((np.asarray(F, np.int64), j))
# warm-up
for F, j in queries[:20]:
    sk.query(F, pairs[j][0], pairs[j][1], x=X[:, j], mode="exact")
lat = {1: [], 2: [], 3: []}
for F, j in queries:
    t0 = time.perf_counter()
    sk.query(F, pairs[j][0], pairs[j][1], x=X[:, j], mode="exact")
    lat[len(F)].append(time.perf_counter() - t0)
res["query_exact_ms"] = {str(k): 1e3 * float(np.median(a)) for k, a in lat.items()}
latt = []
for F, j in queries[:500]:
    t0 = time.perf_counter()
    sk.query(F, pairs[j][0], pairs[j][1], mode="tree")
    latt.append(time.perf_counter() - t0)
res["query_tree_ms"] = 1e3 * float(np.median(latt))
# vectorized all-edge single-failure scan for one pair
s, t = pairs[0]
t0 = time.perf_counter()
fe = G.sw * (X[G.u, 0] - X[G.v, 0])
blk = 1 << 18
cnt = 0
for b0 in range(0, G.m, blk):
    Yb = sk.Y[:, b0:b0 + blk].astype(np.float64)
    cee = np.einsum("ij,ij->j", Yb, Yb)
    with np.errstate(divide="ignore"):
        dlt = fe[b0:b0 + blk] ** 2 / cee
    cnt += len(dlt)
res["scan_all_edges_s"] = time.perf_counter() - t0
res["scan_throughput_edges_per_s"] = cnt / res["scan_all_edges_s"]
print(f"  query exact (ms) {res['query_exact_ms']}  tree {res['query_tree_ms']:.3f}ms  all-edge scan {res['scan_all_edges_s']:.2f}s", flush=True)

# ---- baselines ---------------------------------------------------------------
if args.skip_baselines:
    json.dump(res, open(args.out or os.path.join(os.path.dirname(__file__), "..", "results", f"efficiency_{args.dataset}_d{args.d}.json"), "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in res if k not in ("dataset",)}, default=float), flush=True)
    sys.exit(0)
# online Woodbury: k solves + pair solve per query
wt = []
for F, j in queries[: args.wood_q]:
    t0 = time.perf_counter()
    CFF, _ = woodbury_CFF(G, S, F, tol=1e-10)
    bb = np.zeros(G.n); bb[pairs[j][0]] = 1; bb[pairs[j][1]] = -1
    xx, _ = S.solve(bb, tol=1e-10)
    quad_pinv(CFF, coupling_exact(G, xx, F), len(F))
    wt.append(time.perf_counter() - t0)
res["woodbury_online_ms"] = 1e3 * float(np.median(wt))
# per-column solve time (LODF cache cost per catalog edge)
t0 = time.perf_counter()
edge_cols(G, S, np.arange(min(G.m, 64)), tol=1e-10)
res["t_solve_per_column"] = (time.perf_counter() - t0) / min(G.m, 64)
res["lodf_cache_all_edges_extrapolated_s"] = res["t_solve_per_column"] * G.m
res["lodf_cache_all_edges_bytes_Mcols_n"] = 8.0 * G.m * G.n
# re-solve with a fresh factorization of G \ F
if args.solver == "cholmod" and args.resolve_q > 0:
    rt = []
    for F, j in queries[: args.resolve_q]:
        t0 = time.perf_counter()
        w_new = G.w.copy(); w_new[F] = 0
        resistance_on_subgraph(G, w_new > 0, w_new, pairs[j][0], pairs[j][1])
        rt.append(time.perf_counter() - t0)
    res["resolve_fresh_factor_ms"] = 1e3 * float(np.median(rt))
# original fixed polynomial: degree from trace bound, per-iteration cost
tau = total_stretch(G, T)
res["tau_T"] = tau
for beta in [1e-2, 1e-4]:
    res[f"fixedpoly_degree_beta{beta:g}"] = fixed_poly_degree(1 + tau, beta)
V = G.BW @ rng.standard_normal((G.m, 8))
t0 = time.perf_counter()
apply_fixed_polynomial(G, T, V, 1 + tau, 5)
per_iter_per_probe = (time.perf_counter() - t0) / (5 * 8)
res["fixedpoly_sec_per_iter_per_probe"] = per_iter_per_probe
res["fixedpoly_build_extrapolated_s_beta1e-2"] = per_iter_per_probe * res["fixedpoly_degree_beta0.01"] * args.d + sk.t_lift
print(json.dumps({k: res[k] for k in res if k not in ("dataset",)}, default=float), flush=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
json.dump(res, open(args.out or os.path.join(os.path.dirname(__file__), "..", "results", f"efficiency_{args.dataset}_d{args.d}.json"), "w"), indent=1, default=float)
