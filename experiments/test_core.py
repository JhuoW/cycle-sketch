"""Unit tests of the core library against dense linear algebra."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from core import Graph, SpanningTree, Solver, CycleSketch, TopologyOracle, residual_full

rng = np.random.default_rng(5)


def rand_graph(n, extra, weighted=True):
    perm = rng.permutation(n)
    E = [(perm[i], perm[rng.integers(0, i)]) for i in range(1, n)]
    for _ in range(extra):
        a, b = rng.choice(n, 2, replace=False); E.append((a, b))
    E = np.array(E)
    w = np.exp(rng.uniform(-3, 3, len(E))) if weighted else np.ones(len(E))
    return Graph(n, E[:, 0], E[:, 1], w)


def dense_C(G):
    """Accurate dense references via the grounded Laplacian (vertex 0)."""
    Phi = G.BW.toarray().T[:, 1:]     # m x (n-1)
    Lg = G.L.toarray()[1:, 1:]
    Linv = np.linalg.inv(Lg)
    Lp = np.zeros((G.n, G.n)); Lp[1:, 1:] = Linv   # grounded inverse (valid for b with sum 0)
    return np.eye(G.m) - Phi @ Linv @ Phi.T, Lp


def grounded_pinv(Lfull):
    """Resistance-correct inverse for a possibly disconnected Laplacian:
    ground one vertex per component."""
    import scipy.sparse.csgraph as cg
    nc, lab = cg.connected_components(sp.csr_matrix(np.abs(Lfull) > 0), directed=False)
    n = Lfull.shape[0]; K = np.zeros((n, n))
    for c in range(nc):
        idx = np.flatnonzero(lab == c)
        if len(idx) > 1:
            sub = idx[1:]
            K[np.ix_(sub, sub)] = np.linalg.inv(Lfull[np.ix_(sub, sub)])
    return K


fails = 0
for trial in range(8):
    G = rand_graph(int(rng.integers(20, 60)), int(rng.integers(10, 80)), weighted=trial % 2 == 0)
    C, Lp = dense_C(G)
    for kind in ["mst", "bfs", "random"]:
        T = SpanningTree(G, kind=kind)
        # lift: exact provisional sample -> must equal C g
        g = rng.standard_normal((G.m, 3))
        S = Solver(G, T.root, "cholmod")
        X, _ = S.solve(G.BW @ g)
        y = T.lift((g - G.BW.T @ X)[T.cotree])
        err_lift = np.abs(y - C @ g).max()
        cons = np.abs(G.BW @ y).max()
        # lift with garbage potentials: still conservative; error bound by tree energy
        Xb = X + 0.3 * rng.standard_normal(X.shape)
        yb = T.lift((g - G.BW.T @ Xb)[T.cotree])
        r = residual_full(G, Xb, G.BW @ g)
        _, eta2 = T.route(r)
        dev = ((yb - C @ g) ** 2).sum(0)
        cons_b = np.abs(G.BW @ yb).max()
        # potentials: y^T T r = pi^T r
        rr = rng.standard_normal((G.n, 1)); rr -= rr.mean()
        ych, en = T.route(rr)
        Tr = np.zeros((G.m, 1)); child = T.order[1:]; Tr[T.pedge[child]] = ych[child]
        pi = T.potentials(yb)
        e_pot = np.abs((yb.T @ Tr).ravel() - (pi.T @ rr).ravel()).max()
        # tree energy equals r^T L_T^+ r
        LT = sp.csr_matrix(G.L.shape)
        te = T.tree_edges
        Gt = Graph(G.n, G.u[te], G.v[te], G.w[te])
        Lt = Gt.L.toarray(); Ki = np.linalg.inv(Lt[1:, 1:]); rv = rr[1:, 0] - 0.0
        e_en = abs(en[0] - float(rv @ Ki @ rv)) / en[0]
        # path_flow == T(chi_s - chi_t)
        s, t = rng.choice(G.n, 2, replace=False)
        bb = np.zeros((G.n, 1)); bb[s] = 1; bb[t] = -1
        ych2, _ = T.route(bb)
        Tb = np.zeros(G.m); Tb[T.pedge[child]] = ych2[child, 0]
        e_path = np.abs(T.path_flow(np.arange(G.m), s, t) - Tb).max()
        ok = (err_lift < 1e-9 and cons < 1e-9 and cons_b < 1e-9 and np.all(dev <= eta2 * (1 + 1e-9) + 1e-12)
              and e_pot < 1e-8 and e_en < 1e-8 and e_path < 1e-12)
        if not ok:
            fails += 1
            print("FAIL", trial, kind, err_lift, cons, cons_b, dev, eta2, e_pot, e_en, e_path)

# topology oracle vs BFS
for trial in range(200):
    G = rand_graph(int(rng.integers(8, 25)), int(rng.integers(0, 10)), weighted=False)
    T = SpanningTree(G, kind="random", seed=trial)
    O = TopologyOracle(G, T, seed=trial)
    f = int(rng.integers(1, 5))
    F = rng.choice(G.m, f, replace=False)
    keep = np.ones(G.m, bool); keep[F] = False
    A = sp.coo_matrix((np.ones(keep.sum()), (G.u[keep], G.v[keep])), shape=(G.n, G.n))
    nc, lab = connected_components(A, directed=False)
    s, t = rng.choice(G.n, 2, replace=False)
    extra, sep = O.analyze(F, s, t)
    if extra != nc - 1 or sep != (lab[s] != lab[t]):
        fails += 1
        print("TOPO FAIL", extra, nc - 1, sep, lab[s] != lab[t])

# sketch query vs dense truth (large d -> small error), incl. singular cases
for trial in range(6):
    G = rand_graph(40, 30)
    C, Lp = dense_C(G)
    T = SpanningTree(G, kind="mst")
    S = Solver(G, T.root, "cholmod")
    sk = CycleSketch(G, T, S, d=4000, seed=trial)
    for _ in range(20):
        f = int(rng.integers(1, 4))
        F = rng.choice(G.m, f, replace=False)
        alpha = np.where(rng.random(f) < 0.7, 1.0, rng.uniform(0.1, 0.9, f))
        s, t = rng.choice(G.n, 2, replace=False)
        # dense truth
        w2 = G.w.copy(); w2[F] *= (1 - alpha)
        keep = w2 > 0
        G2 = Graph(G.n, G.u[keep], G.v[keep], w2[keep])
        A = sp.coo_matrix((np.ones(keep.sum()), (G2.u, G2.v)), shape=(G.n, G.n))
        nc, lab = connected_components(A, directed=False)
        b = np.zeros(G.n); b[s] = 1; b[t] = -1
        R0 = b @ Lp @ b
        if lab[s] != lab[t]:
            truth = math.inf
        else:
            R1 = b @ grounded_pinv(G2.L.toarray()) @ b
            truth = R1 - R0
        x = Lp @ b
        for mode in ["exact", "tree"]:
            q = sk.query(F, s, t, alpha=alpha, x=x, mode=mode)
            if math.isinf(truth):
                if not q["separated"]:
                    fails += 1; print("SEP FAIL")
            else:
                rel = abs(q["delta"] - truth) / max(truth, 1e-3 * R0)
                if rel > 0.25 and mode == "exact":
                    fails += 1; print("QUERY FAIL", mode, q, truth, R0)
print("fails:", fails)
