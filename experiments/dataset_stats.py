"""Dataset statistics: sizes, tree stretch (trace bound), predicted degree of the
original fixed-polynomial construction, CHOLMOD/AMG solver costs."""
import sys, os, time, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from graphio import load
from core import Graph, SpanningTree, Solver, residual_full
from baselines import total_stretch, fixed_poly_degree

names = sys.argv[1:]
out = {}
path = os.path.join(os.path.dirname(__file__), "..", "results", "dataset_stats.json")
if os.path.exists(path):
    try:
        out = json.load(open(path))
    except Exception:
        out = {}
for nm in names:
    n, u, v, w = load(nm)
    G = Graph(n, u, v, w, nm)
    rec = {"n": G.n, "m": G.m, "r_cyc": G.m - G.n + 1, "weighted": bool(np.ptp(G.w) > 0)}
    for kind in (["mst", "bfs"] if rec["weighted"] else ["bfs", "random"]):
        T = SpanningTree(G, kind=kind)
        tau = total_stretch(G, T)
        rec[f"tree_{kind}"] = {"build_s": T.build_time, "tau": tau, "avg_stretch": tau / max(1, G.m - G.n + 1),
                               "deg_poly_beta1e-2": fixed_poly_degree(1 + tau, 1e-2),
                               "deg_poly_beta1e-4": fixed_poly_degree(1 + tau, 1e-4)}
        print(nm, kind, rec[f"tree_{kind}"], flush=True)
    T = SpanningTree(G, kind="mst" if rec["weighted"] else "bfs")
    rng = np.random.default_rng(0)
    g = rng.standard_normal((G.m, 4)); rhs = G.BW @ g
    for kind in (sys.argv[0] and os.environ.get("SOLVERS","cholmod,amg").split(",")):
        try:
            t0 = time.time(); S = Solver(G, T.root, kind); ts = time.time() - t0
            t0 = time.time(); X, its = S.solve(rhs, tol=1e-10); tsol = (time.time() - t0) / 4
            r = residual_full(G, X, rhs); _, eta2 = T.route(r)
            info = {"setup_s": ts, "solve_s_per_rhs": tsol, "iters": its, "eta_max": float(np.sqrt(eta2.max()))}
            if kind == "cholmod":
                info["factor_nnz"] = int(S.F.nnz) if hasattr(S.F, "nnz") and not callable(S.F.nnz) else int(S.F.nnz())
            rec[kind] = info
            print(nm, kind, info, flush=True)
            del S
        except Exception as ex:
            rec[kind] = {"error": repr(ex)}
            print(nm, kind, "ERROR", ex, flush=True)
    out[nm] = rec
    json.dump(out, open(path + ".tmp", "w"), indent=1, default=float); os.replace(path + ".tmp", path)
