"""Exact cycle mass C_ee = 1 - w_e R_e for every edge, with parallel workers.

Used ONLY as ground truth for verifying certified screening on graphs where the
single-threaded exhaustive computation (one solve per edge) would take hours.
Each worker builds its own CHOLMOD factor of the same grounded Laplacian and
solves blocks of edge columns at tolerance 1e-12 (direct solve), exactly as
exp_screening.exact_Cdiag does.  The cost reported in the paper for the
exhaustive baseline is the single-threaded per-solve cost times m
(extrapolated), never the wall-clock time of this script.

Output: results/cache/cdiag_<dataset>.npy (float64, length m; bridges -> ~0).
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import multiprocessing as mp

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--workers", type=int, default=24)
ap.add_argument("--block", type=int, default=64)
ap.add_argument("--cpus", default=None, help="comma-separated CPU ids for the workers")
args = ap.parse_args()

_G = _S = None


def _init(cpus):
    global _G, _S
    if cpus:
        try:
            os.sched_setaffinity(0, cpus)
        except OSError:
            pass
    from graphio import load
    from core import Graph, SpanningTree, Solver
    n, u, v, w = load(args.dataset)
    _G = Graph(n, u, v, w, args.dataset)
    T = SpanningTree(_G, kind="mst" if np.ptp(_G.w) > 0 else "bfs")
    _S = Solver(_G, T.root, "cholmod")


def _work(rng_):
    from baselines import edge_cols
    a, b = rng_
    E = np.arange(a, b)
    X, _ = edge_cols(_G, _S, E, tol=1e-12)
    j = np.arange(len(E))
    return a, 1 - _G.sw[E] * (X[_G.u[E], j] - X[_G.v[E], j])


if __name__ == "__main__":
    from graphio import load
    n, u, v, w = load(args.dataset)
    m = len(u)
    cpus = [int(c) for c in args.cpus.split(",")] if args.cpus else None
    chunks = [(a, min(m, a + args.block)) for a in range(0, m, args.block)]
    out = np.full(m, np.nan)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init, initargs=(cpus,)) as pool:
        for i, (a, c) in enumerate(pool.imap_unordered(_work, chunks, chunksize=4)):
            out[a:a + len(c)] = c
            if i % 500 == 0:
                print(f"  {i}/{len(chunks)} blocks, {time.time()-t0:.0f}s", flush=True)
    assert np.isfinite(out).all()
    d = os.path.join(os.path.dirname(__file__), "..", "results", "cache")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"cdiag_{args.dataset}.npy"), out)
    print(f"[{args.dataset}] exact C_ee for m={m} edges in {time.time()-t0:.0f}s with {args.workers} workers; "
          f"min {out.min():.3g} median {np.median(out):.3g}", flush=True)
