"""Graph loading and preprocessing.

All graphs are converted to an undirected, connected (largest component),
positively weighted multigraph stored as arrays (u, v, w) with u != v.
Parallel edges are kept as distinct edges (failures are per edge).
"""
import os, re
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")


def _lcc(n, u, v, w):
    A = sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    big = np.bincount(lab).argmax()
    keep_v = np.flatnonzero(lab == big)
    remap = -np.ones(n, dtype=np.int64)
    remap[keep_v] = np.arange(len(keep_v))
    mask = (lab[u] == big)
    return len(keep_v), remap[u[mask]], remap[v[mask]], w[mask]


def load_snap_tsv(name, dedup=True):
    """GraphChallenge SNAP mirror: 'src dst weight' per line (1-based ids)."""
    path = os.path.join(RAW, name + ".tsv")
    data = np.loadtxt(path, dtype=np.int64, usecols=(0, 1))
    u, v = data[:, 0], data[:, 1]
    mask = u != v
    u, v = u[mask], v[mask]
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    if dedup:  # SNAP graphs: treat as simple undirected graphs
        key = np.unique(np.stack([lo, hi], 1), axis=0)
        lo, hi = key[:, 0], key[:, 1]
    ids, inv = np.unique(np.concatenate([lo, hi]), return_inverse=True)
    u2, v2 = inv[: len(lo)], inv[len(lo):]
    w = np.ones(len(u2))
    return _lcc(len(ids), u2, v2, w)


def load_matpower(name):
    """MATPOWER case: buses and in-service branches; conductance = 1/|x|
    (DC power-flow susceptance), the standard linear network model."""
    txt = open(os.path.join(RAW, name + ".m")).read()

    def block(tag):
        mobj = re.search(r"mpc\.%s\s*=\s*\[(.*?)\];" % tag, txt, re.S)
        rows = []
        for line in mobj.group(1).split("\n"):
            line = line.split("%")[0].strip().rstrip(";").strip()
            if not line:
                continue
            rows.append([float(t) for t in line.replace(";", " ").split()])
        return rows

    bus = block("bus")
    br = block("branch")
    bus_ids = np.array([int(r[0]) for r in bus])
    idx = {b: i for i, b in enumerate(bus_ids)}
    u, v, w = [], [], []
    for r in br:
        f, t, x, status = int(r[0]), int(r[1]), r[3], r[10] if len(r) > 10 else 1.0
        if status == 0 or f == t:
            continue
        x = abs(x)
        if x < 1e-6:
            x = 1e-6  # zero-impedance branches: cap conductance (documented in paper)
        u.append(idx[f]); v.append(idx[t]); w.append(1.0 / x)
    u, v, w = np.array(u), np.array(v), np.array(w)
    return _lcc(len(bus_ids), u, v, w)


DATASETS = {
    # name: (loader, key)
    "PEG2869": ("matpower", "case2869pegase"),
    "PEG9241": ("matpower", "case9241pegase"),
    "PEG13659": ("matpower", "case13659pegase"),
    "ACT25k": ("matpower", "case_ACTIVSg25k"),
    "ACT70k": ("matpower", "case_ACTIVSg70k"),
    "SynUSA": ("matpower", "case_SyntheticUSA"),
    "HepPh": ("snap", "ca-HepPh"),
    "Enron": ("snap", "email-Enron"),
    "Brightkite": ("snap", "loc-brightkite_edges"),
    "Epinions": ("snap", "soc-Epinions1"),
    "Gowalla": ("snap", "loc-gowalla_edges"),
    "Amazon": ("snap", "amazon0601"),
    "RoadPA": ("snap", "roadNet-PA"),
    "RoadTX": ("snap", "roadNet-TX"),
    "RoadCA": ("snap", "roadNet-CA"),
}


def load(name):
    os.makedirs(PROC, exist_ok=True)
    cache = os.path.join(PROC, name + ".npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return int(z["n"]), z["u"], z["v"], z["w"]
    kind, key = DATASETS[name]
    n, u, v, w = load_snap_tsv(key) if kind == "snap" else load_matpower(key)
    np.savez(cache, n=n, u=u.astype(np.int64), v=v.astype(np.int64), w=w.astype(np.float64))
    return n, u, v, w


if __name__ == "__main__":
    import sys, time
    names = sys.argv[1:] or list(DATASETS)
    for nm in names:
        t0 = time.time()
        n, u, v, w = load(nm)
        print(f"{nm:12s} n={n:9d} m={len(u):9d} cyc={len(u)-n+1:9d} wmin={w.min():.3g} wmax={w.max():.3g} ({time.time()-t0:.1f}s)", flush=True)
