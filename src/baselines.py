"""Baselines and exact references.

* CutGram       : same Gaussian solves as CycleSketch, but keeps the cut part
                  Phi x_i (Spielman--Srivastava-type sketch of L^+) and forms
                  C_FF = I - P_FF by subtraction.
* RawCycleGram  : cycle sample g - Phi x~ WITHOUT the fundamental-cycle lift.
* Woodbury      : online exact method, (k+1) solves per query (any solver).
* Resolve       : online, one solve on the modified graph, Delta = R' - R.
* FixedPoly     : the original ERGraph3 construction (tree-preconditioned
                  third-kind Chebyshev polynomial, degree from the trace bound).
* truth()       : high-accuracy reference (CHOLMOD) with exact topology.
"""
import math, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import numba as nb
from core import Graph, Solver, residual_full


class CutGram:
    def __init__(self, G, solver, d, catalog=None, seed=1, tol=1e-10, block=32):
        t0 = time.time()
        self.G, self.d = G, d
        self.catalog = np.arange(G.m) if catalog is None else np.asarray(catalog)
        pos = -np.ones(G.m, np.int64); pos[self.catalog] = np.arange(len(self.catalog)); self.pos = pos
        rng = np.random.default_rng(seed)
        self.Z = np.zeros((d, len(self.catalog)))
        for b0 in range(0, d, block):
            b1 = min(d, b0 + block)
            g = rng.standard_normal((G.m, b1 - b0))
            X, _ = solver.solve(G.BW @ g, tol=tol)
            self.Z[b0:b1] = ((G.BW.T @ X)[self.catalog] / math.sqrt(d)).T
        self.build_time = time.time() - t0

    def CFF(self, F):
        ZF = self.Z[:, self.pos[F]]
        return np.eye(len(F)) - ZF.T @ ZF


class RawCycleGram:
    def __init__(self, G, solver, d, catalog=None, seed=1, tol=1e-10, block=32):
        self.G, self.d = G, d
        self.catalog = np.arange(G.m) if catalog is None else np.asarray(catalog)
        pos = -np.ones(G.m, np.int64); pos[self.catalog] = np.arange(len(self.catalog)); self.pos = pos
        rng = np.random.default_rng(seed)
        self.Y = np.zeros((d, len(self.catalog)))
        for b0 in range(0, d, block):
            b1 = min(d, b0 + block)
            g = rng.standard_normal((G.m, b1 - b0))
            X, _ = solver.solve(G.BW @ g, tol=tol)
            self.Y[b0:b1] = ((g - G.BW.T @ X)[self.catalog] / math.sqrt(d)).T

    def CFF(self, F):
        YF = self.Y[:, self.pos[F]]
        return YF.T @ YF


def quad_pinv(M, z, rank):
    """z^T M^+ z using the top-`rank` eigenpairs of symmetric M (rank fixed by
    topology). Negative/zero retained eigenvalues are reported as inf."""
    M = 0.5 * (M + M.T)
    ev, V = np.linalg.eigh(M)
    ev, V = ev[::-1][:rank], V[:, ::-1][:, :rank]
    if rank == 0:
        return 0.0
    if ev[-1] <= 0:
        return math.inf
    c = V.T @ z
    return float(np.sum(c * c / ev))


def coupling_exact(G, x, F):
    return G.sw[F] * (x[G.u[F]] - x[G.v[F]])


def edge_cols(G, solver, F, tol=1e-12):
    f = len(F)
    Bf = np.zeros((G.n, f))
    np.add.at(Bf, (G.u[F], np.arange(f)), G.sw[F])
    np.add.at(Bf, (G.v[F], np.arange(f)), -G.sw[F])
    X, its = solver.solve(Bf, tol=tol)
    return X, its


def woodbury_CFF(G, solver, F, tol=1e-12):
    X, its = edge_cols(G, solver, F, tol)
    PFF = G.sw[F][:, None] * (X[G.u[F]] - X[G.v[F]])
    return np.eye(len(F)) - 0.5 * (PFF + PFF.T), its


def components_without(G, F):
    keep = np.ones(G.m, bool); keep[F] = False
    A = sp.coo_matrix((np.ones(keep.sum()), (G.u[keep], G.v[keep])), shape=(G.n, G.n))
    return connected_components(A, directed=False)


def resistance_on_subgraph(G, keep_mask, w_new, s, t):
    """R(s,t) on the graph with edges keep_mask and conductances w_new, by a
    fresh CHOLMOD factorization of the component containing s."""
    from sksparse.cholmod import cho_factor
    u, v, w = G.u[keep_mask], G.v[keep_mask], w_new[keep_mask]
    A = sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(G.n, G.n))
    nc, lab = connected_components(A, directed=False)
    if lab[s] != lab[t]:
        return math.inf
    nodes = np.flatnonzero(lab == lab[s])
    remap = -np.ones(G.n, np.int64); remap[nodes] = np.arange(len(nodes))
    msk = lab[u] == lab[s]
    H = Graph(len(nodes), remap[u[msk]], remap[v[msk]], w[msk])
    root = remap[t]
    keep = np.ones(H.n, bool); keep[root] = False
    Lg = H.L[keep][:, keep].tocsc()
    Fc = cho_factor(Lg)
    b = np.zeros(H.n); b[remap[s]] = 1.0
    x = Fc.solve(b[keep])
    return float(x[np.flatnonzero(keep) == remap[s]][0]) if False else float(x[np.searchsorted(np.flatnonzero(keep), remap[s])])


def truth(G, solver, F, s, t, alpha=None, x=None, R0=None, refactor_if_mu_below=1e-7):
    """High-accuracy Delta_F(s,t). Uses Woodbury with CHOLMOD columns and the
    exact topological rank; if the supported failure matrix is very
    ill-conditioned, recomputes R' by refactorizing the modified graph."""
    F = np.asarray(F, dtype=np.int64); f = len(F)
    alpha = np.ones(f) if alpha is None else np.asarray(alpha, float)
    full = F[alpha >= 1]
    nc, lab = components_without(G, full) if len(full) else (1, np.zeros(G.n, int))
    if lab[s] != lab[t]:
        return {"delta": math.inf, "R0": R0, "mu": 0.0, "rank": None}
    if x is None:
        b = np.zeros(G.n); b[s] = 1; b[t] = -1
        x, _ = solver.solve(b)
    if R0 is None:
        R0 = float(x[s] - x[t])
    CFF, _ = woodbury_CFF(G, solver, F)
    D = np.sqrt(alpha)
    M = np.eye(f) - np.diag(alpha) + D[:, None] * CFF * D[None, :]
    z = D * coupling_exact(G, x, F)
    rank = f - (nc - 1)
    # exact kernel from topology: component indicators -> B^T O
    if nc > 1:
        comps = np.unique(lab[np.concatenate([G.u[full], G.v[full]])])
        O = []
        for c in comps:
            o = (lab == c).astype(float)
            O.append(D * G.sw[F] * (o[G.u[F]] - o[G.v[F]]))
        N = np.array(O).T
        Q, _ = np.linalg.qr(N)
        # orthonormal complement of range(N) restricted to rank
        U, sv, Vt = np.linalg.svd(N, full_matrices=True)
        r_n = int(np.sum(sv > 1e-10 * sv.max()))
        W = U[:, r_n:]
    else:
        W = np.eye(f)
    Ms = W.T @ M @ W
    zs = W.T @ z
    ev = np.linalg.eigvalsh(0.5 * (Ms + Ms.T)) if Ms.size else np.array([1.0])
    mu = float(ev.min()) if Ms.size else 1.0
    if Ms.size == 0:
        delta = 0.0
    else:
        delta = float(zs @ np.linalg.solve(Ms, zs))
    if mu < refactor_if_mu_below:
        w_new = G.w.copy(); w_new[F] *= (1 - alpha)
        keep = w_new > 0
        R1 = resistance_on_subgraph(G, keep, w_new, s, t)
        delta = R1 - R0
    return {"delta": delta, "R0": R0, "mu": mu, "rank": rank}


# ----------------------------------------------------------------------------
# Original ERGraph3 fixed polynomial construction
# ----------------------------------------------------------------------------


@nb.njit(cache=True)
def _lca_res(u, v, parent, depth, rdepth, up, LOG):
    out = np.empty(u.shape[0])
    for i in range(u.shape[0]):
        a, b = u[i], v[i]
        if depth[a] < depth[b]:
            a, b = b, a
        diff = depth[a] - depth[b]
        k = 0
        while diff > 0:
            if diff & 1:
                a = up[k, a]
            diff >>= 1; k += 1
        if a != b:
            for k in range(LOG - 1, -1, -1):
                if up[k, a] != up[k, b]:
                    a = up[k, a]; b = up[k, b]
            a = parent[a]
        out[i] = rdepth[u[i]] + rdepth[v[i]] - 2 * rdepth[a]
    return out


def tree_stretch_resistances(tree, us, vs):
    """R_T(u,v) for many pairs via binary lifting."""
    n = len(tree.parent)
    if not hasattr(tree, "_up"):
        depth = np.zeros(n, np.int64)
        for c in tree.order[1:]:
            depth[c] = depth[tree.parent[c]] + 1
        LOG = max(1, int(depth.max()).bit_length())
        up = np.empty((LOG, n), np.int64); up[0] = tree.parent
        for k in range(1, LOG):
            up[k] = up[k - 1][up[k - 1]]
        tree._up, tree._depth, tree._LOG = up, depth, LOG
    return _lca_res(np.asarray(us, np.int64), np.asarray(vs, np.int64), tree.parent, tree._depth, tree.rdepth, tree._up, tree._LOG)


def total_stretch(G, tree):
    N = tree.cotree
    RT = tree_stretch_resistances(tree, G.u[N], G.v[N])
    return float(np.sum(G.w[N] * RT))


def tree_solve(tree, R):
    """L_T^{-1} r (grounded at tree root): potentials of the tree routing."""
    S = tree.subtree_sums(R)
    S[tree.root, :] = 0.0
    n = tree.G.n
    from core import _potentials_from_drops
    drop = np.zeros_like(S)
    child = tree.order[1:]
    drop[child] = S[child] / tree.wpe[child][:, None]   # potential(child) - potential(parent) = flow/w
    P = np.zeros_like(S)
    _potentials_from_drops(tree.order, tree.parent, drop, P)
    return P


def fixed_poly_degree(Lam, beta_star):
    a = math.asinh(1 / math.sqrt(Lam - 1))
    return max(0, math.ceil(0.5 * (math.asinh(1 / math.sqrt(beta_star)) / a - 1)))


def apply_fixed_polynomial(G, tree, V, Lam, degree):
    """H V for H = p_{deg-1}(L_T^{-1} L) L_T^{-1} (grounded at tree root)."""
    if degree == 0:
        return np.zeros_like(V)
    a = 2 / (Lam - 1); c = -(Lam + 1) / (Lam - 1); nu = 2 * c - 1
    V = V.copy(); V[tree.root] = 0.0
    hp = np.zeros_like(V)
    hc = (-2 * a / nu) * tree_solve(tree, V)
    for j in range(1, degree):
        nun = 2 * c - 1 / nu
        res = V - G.L @ hc
        res[tree.root] = 0.0
        hn = (2 * c / nun) * hc - hp / (nu * nun) - (2 * a / nun) * tree_solve(tree, res)
        hp, hc = hc, hn
        nu = nun
    return hc


class FixedPolySolver:
    """Wraps the original construction as a 'solver' for CycleSketch."""

    def __init__(self, G, tree, beta_star=1e-2, Lam=None, max_degree=None):
        t0 = time.time()
        self.G, self.tree = G, tree
        self.tau = total_stretch(G, tree) - 0.0
        # tau_T = sum_cotree w_e R_T(e)  (tree edges contribute exactly n-1 to the trace)
        self.Lam = 1 + self.tau if Lam is None else Lam
        self.degree = fixed_poly_degree(self.Lam, beta_star)
        if max_degree is not None:
            self.degree = min(self.degree, max_degree)
        self.setup_time = time.time() - t0

    def solve(self, B, tol=None):
        X = apply_fixed_polynomial(self.G, self.tree, np.asarray(B, float), self.Lam, self.degree)
        return X, [self.degree] * B.shape[1]
