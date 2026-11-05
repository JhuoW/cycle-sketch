"""EXP-H: GF(2) cycle-space topology oracle vs. BFS.

Scenarios that frequently disconnect: remove k edges around low-degree
vertices (isolating them or cutting short chains), plus random k-sets.
Checks (i) #extra components and (ii) s-t separation against BFS, and times both.
"""
import sys, os, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from graphio import load
from core import Graph, SpanningTree, TopologyOracle

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--nq", type=int, default=3000)
ap.add_argument("--bfs_q", type=int, default=200)
args = ap.parse_args()
n, u, v, w = load(args.dataset)
G = Graph(n, u, v, w, args.dataset)
T = SpanningTree(G, kind="mst" if np.ptp(G.w) > 0 else "bfs")
t0 = time.time(); O = TopologyOracle(G, T, seed=5); t_build = time.time() - t0
deg = np.bincount(np.concatenate([G.u, G.v]), minlength=G.n)
inc = sp.csr_matrix((np.ones(2 * G.m), (np.concatenate([G.u, G.v]), np.concatenate([np.arange(G.m)] * 2))), shape=(G.n, G.m))
rng = np.random.default_rng(1)
lowdeg = np.flatnonzero((deg >= 2) & (deg <= 4))
Q = []
for i in range(args.nq):
    if i % 2 == 0 and len(lowdeg):
        x = int(rng.choice(lowdeg))
        E = inc.indices[inc.indptr[x]:inc.indptr[x + 1]]
        k = min(len(E), 4)
        F = rng.choice(E, k, replace=False)
        s = x if rng.random() < 0.5 else int(rng.integers(G.n))
    else:
        k = int(rng.integers(1, 5))
        F = rng.choice(G.m, k, replace=False); s = int(rng.integers(G.n))
    t = int(rng.integers(G.n))
    Q.append((np.asarray(F, np.int64), s, t))
# oracle timing on all queries
t0 = time.perf_counter()
ans = [O.analyze(F, s, t) for F, s, t in Q]
t_oracle = (time.perf_counter() - t0) / len(Q)
# BFS ground truth on a subset (each BFS is O(n+m))
mism_c = mism_s = 0; tb = []
nsep = 0; ndisc = 0
for i in range(min(args.bfs_q, len(Q))):
    F, s, t = Q[i]
    t0 = time.perf_counter()
    keep = np.ones(G.m, bool); keep[F] = False
    A = sp.coo_matrix((np.ones(keep.sum()), (G.u[keep], G.v[keep])), shape=(G.n, G.n))
    nc, lab = connected_components(A, directed=False)
    tb.append(time.perf_counter() - t0)
    extra, sep = ans[i]
    mism_c += int(extra != nc - 1); mism_s += int(sep != (lab[s] != lab[t]))
    nsep += int(lab[s] != lab[t]); ndisc += int(nc > 1)
res = {"dataset": args.dataset, "n": G.n, "m": G.m, "t_label_build_s": t_build, "oracle_us_per_query": 1e6 * t_oracle,
       "bfs_ms_per_query": 1e3 * float(np.median(tb)), "checked": min(args.bfs_q, len(Q)),
       "mismatch_components": mism_c, "mismatch_separation": mism_s, "n_separated": nsep, "n_disconnected": ndisc}
print(json.dumps(res), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", f"topology_{args.dataset}.json"), "w"), indent=1)
