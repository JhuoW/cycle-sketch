"""CHOLMOD factor size (nonzeros, peak RSS) of the grounded Laplacian for each dataset,
in the current environment.  Not a timing experiment."""
import sys, os, json, time, resource
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from graphio import load
from core import Graph, SpanningTree, Solver

path = os.path.join(os.path.dirname(__file__), "..", "results", "factor_sizes.json")
out = json.load(open(path)) if os.path.exists(path) else {}
for ds in sys.argv[1:]:
    n, u, v, w = load(ds)
    G = Graph(n, u, v, w, ds)
    T = SpanningTree(G, kind="mst" if np.ptp(G.w) > 0 else "bfs")
    t0 = time.time(); S = Solver(G, T.root, "cholmod"); ts = time.time() - t0
    nnz = S.F.nnz if not callable(getattr(S.F, "nnz", None)) else S.F.nnz()
    out[ds] = {"factor_nnz": int(nnz), "factor_mb_12B_per_nz": 12 * int(nnz) / 1e6, "setup_s_untimed": ts,
               "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    print(ds, out[ds], flush=True)
    json.dump(out, open(path, "w"), indent=1)
    del S
