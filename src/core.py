"""Core library: graphs, spanning trees, tree routing/lifting, Laplacian solvers,
the certified cycle-space sketch (CycleSketch), the GF(2) topology oracle, and
what-if effective-resistance queries.

Conventions
-----------
* Graph: n vertices, m edges (u_e, v_e, w_e), edge e oriented u_e -> v_e.
* Normalized edge coordinates: a flow y in R^m has physical value sqrt(w_e) y_e.
  The (full) weighted incidence is Phi^T = B W^{1/2} (n x m); L = B W B^T.
* Cycle space = ker(B W^{1/2}); C = orthogonal projector onto it.
* Grounding: vertex `root` (the spanning-tree root) has potential 0.
"""
import math, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
import numba as nb

# ----------------------------------------------------------------------------
# Graph container
# ----------------------------------------------------------------------------


class Graph:
    def __init__(self, n, u, v, w, name=""):
        self.n, self.m = int(n), len(u)
        self.u = np.ascontiguousarray(u, dtype=np.int64)
        self.v = np.ascontiguousarray(v, dtype=np.int64)
        self.w = np.ascontiguousarray(w, dtype=np.float64)
        self.sw = np.sqrt(self.w)
        self.name = name
        e = np.arange(self.m)
        # signed, weighted incidence B W^{1/2}: n x m
        self.BW = sp.csr_matrix((np.concatenate([self.sw, -self.sw]),
                                 (np.concatenate([self.u, self.v]), np.concatenate([e, e]))),
                                shape=(self.n, self.m))
        self.L = sp.csr_matrix((np.concatenate([self.w, self.w, -self.w, -self.w]),
                                (np.concatenate([self.u, self.v, self.u, self.v]),
                                 np.concatenate([self.u, self.v, self.v, self.u]))),
                               shape=(self.n, self.n))
        self.L.sum_duplicates()

    def adjacency(self):
        """Simple adjacency storing (edge id + 1); for parallel edges the
        highest-conductance edge id is kept."""
        lo, hi = np.minimum(self.u, self.v), np.maximum(self.u, self.v)
        idx = np.lexsort((-self.w, hi, lo))
        first = np.ones(self.m, bool)
        first[1:] = (lo[idx][1:] != lo[idx][:-1]) | (hi[idx][1:] != hi[idx][:-1])
        e = idx[first]
        return sp.csr_matrix((np.concatenate([e, e]) + 1,
                              (np.concatenate([lo[e], hi[e]]), np.concatenate([hi[e], lo[e]]))),
                             shape=(self.n, self.n))


# ----------------------------------------------------------------------------
# Spanning tree and tree kernels
# ----------------------------------------------------------------------------

@nb.njit(cache=True)
def _subtree_sum_inplace(order, parent, X):
    # X: n x d; process vertices in reverse BFS order, add to parent
    for i in range(order.shape[0] - 1, 0, -1):
        c = order[i]
        p = parent[c]
        for j in range(X.shape[1]):
            X[p, j] += X[c, j]


@nb.njit(cache=True)
def _subtree_xor_inplace(order, parent, h):
    for i in range(order.shape[0] - 1, 0, -1):
        c = order[i]
        h[parent[c]] ^= h[c]


@nb.njit(cache=True)
def _potentials_from_drops(order, parent, drop, P):
    # P[root]=0; P[c] = P[parent[c]] + drop[c]   (drop: n x d)
    for j in range(P.shape[1]):
        P[order[0], j] = 0.0
    for i in range(1, order.shape[0]):
        c = order[i]
        p = parent[c]
        for j in range(P.shape[1]):
            P[c, j] = P[p, j] + drop[c, j]


@nb.njit(cache=True)
def _euler(n, root, child_ptr, child_idx):
    tin = np.empty(n, np.int64); tout = np.empty(n, np.int64)
    stack = np.empty(n, np.int64); it = np.zeros(n, np.int64)
    sp_ = 0; stack[0] = root; timer = 0; tin[root] = timer; timer += 1
    while sp_ >= 0:
        v = stack[sp_]
        k = child_ptr[v] + it[v]
        if k < child_ptr[v + 1]:
            it[v] += 1
            c = child_idx[k]
            tin[c] = timer; timer += 1
            sp_ += 1; stack[sp_] = c
        else:
            tout[v] = timer; timer += 1
            sp_ -= 1
    return tin, tout


class SpanningTree:
    """Rooted spanning tree with O(n) routing/lifting kernels."""

    def __init__(self, G, kind="auto", root=None, seed=0):
        t0 = time.time()
        self.G = G
        n, m = G.n, G.m
        if kind == "auto":
            kind = "mst" if np.ptp(G.w) > 0 else "bfs"
        self.kind = kind
        if root is None:
            deg = np.bincount(np.concatenate([G.u, G.v]), minlength=n)
            root = int(np.argmax(deg))
        if kind == "bfs":
            adj = G.adjacency()
            order, pred = breadth_first_order(adj, root, directed=False, return_predecessors=True)
            # recover parent edge ids
            tree_edges = self._edges_from_pred(adj, order, pred)
        elif kind in ("mst", "random"):
            if kind == "mst":
                key = 1.0 / G.w
            else:
                key = np.random.default_rng(seed).random(m) + 1e-12
            lo, hi = np.minimum(G.u, G.v), np.maximum(G.u, G.v)
            # dedupe parallel edges: keep the best (smallest key) per pair
            idx = np.lexsort((key, hi, lo))
            first = np.ones(m, bool)
            first[1:] = (lo[idx][1:] != lo[idx][:-1]) | (hi[idx][1:] != hi[idx][:-1])
            keep = idx[first]
            Mk = sp.csr_matrix((key[keep], (lo[keep], hi[keep])), shape=(n, n))
            T = minimum_spanning_tree(Mk).tocoo()
            pair_to_edge = {}
            for e in keep:
                pair_to_edge[(int(lo[e]), int(hi[e]))] = int(e)
            tree_edges = np.array([pair_to_edge[(min(a, b), max(a, b))] for a, b in zip(T.row, T.col)], dtype=np.int64)
        else:
            raise ValueError(kind)
        assert len(tree_edges) == n - 1, (len(tree_edges), n)
        self.tree_edges = np.sort(tree_edges)
        is_tree = np.zeros(m, bool); is_tree[self.tree_edges] = True
        self.is_tree = is_tree
        self.cotree = np.flatnonzero(~is_tree)
        # root the tree
        te = self.tree_edges
        tadj = sp.csr_matrix((np.concatenate([te, te]) + 1,
                              (np.concatenate([G.u[te], G.v[te]]), np.concatenate([G.v[te], G.u[te]]))), shape=(n, n))
        order, pred = breadth_first_order(tadj, root, directed=False, return_predecessors=True)
        assert len(order) == n
        self.root = root
        self.order = order.astype(np.int64)
        parent = pred.astype(np.int64); parent[root] = root
        self.parent = parent
        pe = np.full(n, -1, np.int64)
        child = order[1:]
        pe[child] = np.asarray(tadj[child, parent[child]]).ravel().astype(np.int64) - 1
        self.pedge = pe
        # orientation: +1 if tree edge is oriented child -> parent
        sgn = np.zeros(n)
        sgn[child] = np.where(G.u[pe[child]] == child, 1.0, -1.0)
        self.sgn = sgn
        self.wpe = np.ones(n); self.wpe[child] = G.w[pe[child]]
        depth = np.zeros(n, np.int64)
        for c in order[1:]:
            pass
        # Euler tour (ancestor tests)
        cnt = np.bincount(parent[child], minlength=n)
        ptr = np.zeros(n + 1, np.int64); ptr[1:] = np.cumsum(cnt)
        cidx = np.empty(n - 1, np.int64); fill = ptr[:-1].copy()
        for c in child:  # python loop once (n) -- acceptable
            p = parent[c]; cidx[fill[p]] = c; fill[p] += 1
        self.tin, self.tout = _euler(n, root, ptr, cidx)
        # child vertex of each tree edge
        self.child_of_edge = -np.ones(m, np.int64); self.child_of_edge[pe[child]] = child
        # tree-path resistance from root: rT[v] = sum 1/w along root path
        drop = np.zeros((n, 1)); drop[child, 0] = 1.0 / self.wpe[child]
        P = np.zeros((n, 1)); _potentials_from_drops(self.order, self.parent, drop, P)
        self.rdepth = P[:, 0]
        self.build_time = time.time() - t0

    @staticmethod
    def _edges_from_pred(adj, order, pred):
        child = order[1:]
        return (np.asarray(adj[child, pred[child]]).ravel() - 1).astype(np.int64)

    # --- kernels -----------------------------------------------------------
    def subtree_sums(self, X):
        X = np.array(X, dtype=np.float64, copy=True, order="C")
        if X.ndim == 1:
            X = X[:, None]
        _subtree_sum_inplace(self.order, self.parent, X)
        return X

    def route(self, R):
        """Tree routing T r for demand columns R (n x d, net outflow).
        Returns normalized tree-edge values indexed by child vertex (n x d)
        and the tree energies r^T L_T^{-1} r per column."""
        S = self.subtree_sums(R)
        S[self.root, :] = 0.0
        y_child = self.sgn[:, None] * S / np.sqrt(self.wpe)[:, None]
        energy = (S ** 2 / self.wpe[:, None]).sum(0)
        return y_child, energy

    def lift(self, Vcot):
        """Given provisional cotree values (|N| x d, normalized), return the full
        conservative edge vectors y (m x d) with y_N = Vcot and tree values set so
        that B W^{1/2} y = 0 (fundamental-cycle lift J)."""
        G = self.G
        N = self.cotree
        d = Vcot.shape[1]
        phys = Vcot * G.sw[N][:, None]
        div = np.zeros((G.n, d))
        np.add.at(div, G.u[N], phys)
        np.add.at(div, G.v[N], -phys)
        S = self.subtree_sums(div)          # net cotree outflow from each subtree
        S[self.root, :] = 0.0
        # tree flow child->parent must be -S[c]
        y = np.empty((G.m, d))
        y[N] = Vcot
        child = self.order[1:]
        y[self.pedge[child]] = (self.sgn[child, None] * (-S[child])) / np.sqrt(self.wpe[child])[:, None]
        return y

    def potentials(self, Y):
        """Tree potentials pi = T^T y for edge vectors Y (m x d): pi[root]=0 and
        sqrt(w_e)(pi_u - pi_v) = y_e on tree edges."""
        n = self.G.n
        child = self.order[1:]
        drop = np.zeros((n, Y.shape[1]))
        drop[child] = self.sgn[child, None] * Y[self.pedge[child]] / np.sqrt(self.wpe[child])[:, None]
        P = np.zeros((n, Y.shape[1]))
        _potentials_from_drops(self.order, self.parent, drop, P)
        return P

    def on_path(self, e, s, t):
        """1 if edge e lies on the tree path s--t (vectorized over e)."""
        c = self.child_of_edge[e]
        ok = c >= 0
        cc = np.where(ok, c, 0)
        ins = (self.tin[cc] <= self.tin[s]) & (self.tout[s] <= self.tout[cc])
        int_ = (self.tin[cc] <= self.tin[t]) & (self.tout[t] <= self.tout[cc])
        return ok & (ins != int_)

    def path_flow(self, e, s, t):
        """Normalized tree-path routing of chi_s - chi_t evaluated on edges e."""
        c = self.child_of_edge[e]
        ok = c >= 0
        cc = np.where(ok, c, 0)
        ins = ((self.tin[cc] <= self.tin[s]) & (self.tout[s] <= self.tout[cc])).astype(float)
        int_ = ((self.tin[cc] <= self.tin[t]) & (self.tout[t] <= self.tout[cc])).astype(float)
        flow = (ins - int_) * ok          # flow child->parent = [s in sub] - [t in sub]
        return self.sgn[cc] * flow / np.sqrt(self.wpe[cc])

    def lca_resistance(self, s, t):
        """Tree resistance R_T(s,t) (= ||T b||^2) via Euler tour + root distances."""
        # climb is avoided: R_T = rdepth[s] + rdepth[t] - 2 rdepth[lca]; compute lca by
        # walking up from s (used only in evaluation, not in timed queries)
        a, b = int(s), int(t)
        anc = set()
        x = a
        while True:
            anc.add(x)
            if x == self.root:
                break
            x = int(self.parent[x])
        x = b
        while x not in anc:
            x = int(self.parent[x])
        return self.rdepth[a] + self.rdepth[b] - 2 * self.rdepth[x]


# ----------------------------------------------------------------------------
# Laplacian solvers (grounded at the tree root)
# ----------------------------------------------------------------------------


class Solver:
    """Grounded Laplacian solver. kind in {'cholmod','amg'}.
    solve(Bfull, tol) takes full n x d RHS (root row ignored) and returns full
    potentials with x[root]=0 and the number of iterations used."""

    def __init__(self, G, root, kind="cholmod"):
        t0 = time.time()
        self.G, self.root, self.kind = G, root, kind
        keep = np.ones(G.n, bool); keep[root] = False
        self.keep = np.flatnonzero(keep)
        Lg = G.L[self.keep][:, self.keep].tocsc()
        self.Lg = Lg
        if kind == "cholmod":
            from sksparse.cholmod import cho_factor
            self.F = cho_factor(Lg)
        elif kind == "amg":
            import pyamg
            self.ml = pyamg.smoothed_aggregation_solver(Lg.tocsr(), symmetry="hermitian", max_coarse=500)
        elif kind == "jacobi":
            self.Lcsr = Lg.tocsr()
            self.dinv = 1.0 / self.Lcsr.diagonal()
        else:
            raise ValueError(kind)
        self.setup_time = time.time() - t0

    def solve(self, Bfull, tol=1e-10, maxiter=500):
        Bfull = np.asarray(Bfull, dtype=np.float64)
        one = Bfull.ndim == 1
        if one:
            Bfull = Bfull[:, None]
        rhs = Bfull[self.keep]
        X = np.zeros((self.G.n, Bfull.shape[1]))
        its = []
        if self.kind == "cholmod":
            X[self.keep] = self.F.solve(rhs)
            its = [0] * Bfull.shape[1]
        elif self.kind == "jacobi":
            Xk, its = block_pcg(self.Lcsr, rhs, self.dinv, tol=tol, maxiter=maxiter * 20)
            X[self.keep] = Xk
        else:
            for j in range(rhs.shape[1]):
                res = []
                x = self.ml.solve(rhs[:, j], tol=tol, accel="cg", maxiter=maxiter, residuals=res)
                X[self.keep, j] = x
                its.append(len(res) - 1)
        return (X[:, 0] if one else X), its


def block_pcg(A, B, dinv, tol=1e-10, maxiter=10000):
    """Independent (column-wise) Jacobi-preconditioned CG on all columns at
    once; each column stops at relative residual `tol`."""
    X = np.zeros_like(B)
    R = B.copy()
    Z = dinv[:, None] * R
    P = Z.copy()
    rz = np.einsum("ij,ij->j", R, Z)
    bn = np.linalg.norm(B, axis=0); bn[bn == 0] = 1
    active = np.ones(B.shape[1], bool)
    its = np.zeros(B.shape[1], int)
    for it in range(maxiter):
        a_idx = np.flatnonzero(active)
        if len(a_idx) == 0:
            break
        AP = A @ P[:, a_idx]
        alpha = rz[a_idx] / np.einsum("ij,ij->j", P[:, a_idx], AP)
        X[:, a_idx] += P[:, a_idx] * alpha
        R[:, a_idx] -= AP * alpha
        its[a_idx] += 1
        rn = np.linalg.norm(R[:, a_idx], axis=0) / bn[a_idx]
        done = rn < tol
        Zs = dinv[:, None] * R[:, a_idx]
        rz_new = np.einsum("ij,ij->j", R[:, a_idx], Zs)
        beta = rz_new / rz[a_idx]
        P[:, a_idx] = Zs + P[:, a_idx] * beta
        rz[a_idx] = rz_new
        active[a_idx[done]] = False
    return X, [int(i) for i in its]


def residual_full(G, X, Bfull):
    """Residual Bfull - L X with the root equation included (the root absorbs
    the imbalance; tree routing ignores the root entry)."""
    return np.asarray(Bfull) - G.L @ X


# ----------------------------------------------------------------------------
# GF(2) cycle-space labels: exact topology w.h.p.
# ----------------------------------------------------------------------------


class TopologyOracle:
    """64-bit random cycle-space labels (Pritchard--Thurimella style).
    A nonempty edge set S is a cut iff XOR of its labels is 0 (w.p. 1-2^-64
    for non-cuts)."""

    def __init__(self, G, tree, seed=0):
        rng = np.random.default_rng(seed)
        m, n = G.m, G.n
        lab = np.zeros(m, dtype=np.uint64)
        N = tree.cotree
        lab[N] = rng.integers(0, 2 ** 63, size=len(N), dtype=np.uint64) * np.uint64(2) + \
            rng.integers(0, 2, size=len(N), dtype=np.uint64)
        h = np.zeros(n, dtype=np.uint64)
        np.bitwise_xor.at(h, G.u[N], lab[N])
        np.bitwise_xor.at(h, G.v[N], lab[N])
        _subtree_xor_inplace(tree.order, tree.parent, h)
        child = tree.order[1:]
        lab[tree.pedge[child]] = h[child]
        self.lab = lab
        self.tree = tree

    def analyze(self, F_full, s=None, t=None):
        """Return (num_extra_components, separated_flag).
        F_full: edges that are completely removed."""
        F_full = np.asarray(F_full, dtype=np.int64)
        f = len(F_full)
        if f == 0:
            return 0, False
        rows = [int(x) for x in self.lab[F_full]]
        par = [0] * f
        if s is not None:
            par = [int(b) for b in self.tree.on_path(F_full, s, t)]
        # Gaussian elimination over GF(2) on 65-bit rows (label | parity bit)
        vecs = [(rows[i] << 1) | par[i] for i in range(f)]
        basis = []  # reduced rows with pivots on label bits
        rank_lab = 0
        sep = False
        for vec in vecs:
            x = vec
            for b in basis:
                if x ^ b < x:
                    x ^= b
            if (x >> 1) != 0:
                basis.append(x)
                basis.sort(reverse=True)
                rank_lab += 1
            else:
                # label part reduced to zero: a cut combination; parity says s|t
                if x & 1:
                    sep = True
        extra = f - rank_lab
        return extra, sep


# ----------------------------------------------------------------------------
# The certified cycle sketch
# ----------------------------------------------------------------------------


class CycleSketch:
    """Index: d lifted Gaussian cycle samples restricted to a catalog, tree
    potentials of the samples, per-probe certificates eta_i."""

    def __init__(self, G, tree, solver, d, catalog=None, seed=1, tol=1e-10, block=32,
                 lift=True, store_dtype=np.float64, keep_potentials=True, verbose=False,
                 spectral_certificate=True):
        t0 = time.time()
        self.G, self.tree, self.d, self.lift_on = G, tree, d, lift
        self.catalog = np.arange(G.m) if catalog is None else np.asarray(catalog)
        pos = -np.ones(G.m, np.int64); pos[self.catalog] = np.arange(len(self.catalog))
        self.pos = pos
        rng = np.random.default_rng(seed)
        self.Y = np.zeros((d, len(self.catalog)), dtype=store_dtype)
        self.Pi = np.zeros((d, G.n), dtype=store_dtype) if keep_potentials else None
        self.eta2 = np.zeros(d)
        self.iters = []
        t_solve = 0.0; t_lift = 0.0
        Psi = np.zeros((G.n, d), dtype=np.float32) if spectral_certificate else None
        for b0 in range(0, d, block):
            b1 = min(d, b0 + block)
            g = rng.standard_normal((G.m, b1 - b0))
            rhs = G.BW @ g
            ts = time.time()
            X, its = solver.solve(rhs, tol=tol)
            t_solve += time.time() - ts
            self.iters += list(its)
            tl = time.time()
            Vprov = g - G.BW.T @ X                      # g - Phi x~ (normalized)
            if lift:
                Y = tree.lift(Vprov[tree.cotree])
            else:
                Y = Vprov
            r = residual_full(G, X, rhs)
            ych, eta2 = tree.route(r)
            self.eta2[b0:b1] = eta2
            if Psi is not None:
                Psi[:, b0:b1] = ych
            self.Y[b0:b1] = (Y[self.catalog] / math.sqrt(d)).T
            if keep_potentials:
                self.Pi[b0:b1] = (tree.potentials(Y) / math.sqrt(d)).T
            t_lift += time.time() - tl
            if verbose:
                print(f"  probes {b1}/{d}  solve {t_solve:.1f}s lift {t_lift:.1f}s", flush=True)
        self.xi_frob = math.sqrt(self.eta2.mean())     # Frobenius certificate
        if Psi is not None:
            # spectral certificate: xi = sqrt(lambda_max(Gamma)/d), Gamma_ij = r_i^T L_T^+ r_j
            Gam = np.zeros((d, d))
            for c0 in range(0, G.n, 1 << 20):
                P_ = Psi[c0:c0 + (1 << 20)].astype(np.float64)
                Gam += P_.T @ P_
            lam = float(np.linalg.eigvalsh(Gam)[-1])
            self.xi = math.sqrt(max(lam, 0.0) / d)
            del Psi
        else:
            self.xi = self.xi_frob
        self.build_time = time.time() - t0
        self.t_solve, self.t_lift = t_solve, t_lift
        self.topo = TopologyOracle(G, tree, seed=seed + 7)

    def nbytes(self):
        return self.Y.nbytes + (self.Pi.nbytes if self.Pi is not None else 0) + self.topo.lab.nbytes

    # ------------------------------------------------------------------
    def coupling(self, F, s, t, x=None, mode="tree", r=None):
        """Normalized flow of pair (s,t) on edges F.
        mode='exact': x is the exact potential (f_F = Phi x on F).
        mode='tree' : routing p = tree path, corrected by the sketch.
        mode='approx': x approximate potential with residual r (full vector)."""
        G = self.G
        F = np.asarray(F)
        if mode == "exact":
            return G.sw[F] * (x[G.u[F]] - x[G.v[F]]), 0.0
        if mode == "tree":
            pF = self.tree.path_flow(F, s, t)
            Yp = self.Pi[:, s].astype(float) - self.Pi[:, t].astype(float)
            excess_bound = self.tree.lca_resistance(s, t)  # ||Cp||^2 <= R_T(s,t)
        elif mode == "approx":
            ycF, _ = self.tree.route(r[:, None])
            treeF = np.where(self.tree.child_of_edge[F] >= 0, ycF[np.maximum(self.tree.child_of_edge[F], 0), 0], 0.0)
            pF = G.sw[F] * (x[G.u[F]] - x[G.v[F]]) + treeF
            Yp = self.Pi.astype(float) @ r
            _, e2 = self.tree.route(r[:, None]); excess_bound = float(e2[0])
        YF = self.Y[:, self.pos[F]].astype(float)
        return pF - YF.T @ Yp, excess_bound

    def query(self, F, s, t, alpha=None, x=None, mode="tree", r=None, debias=False, fF=None):
        """What-if query. Returns dict with delta_hat, separated flag, rank."""
        G = self.G
        F = np.asarray(F, dtype=np.int64)
        f = len(F)
        alpha = np.ones(f) if alpha is None else np.asarray(alpha, float)
        full = F[alpha >= 1.0]
        extra, sep = self.topo.analyze(full, s, t)
        if sep:
            return {"delta": math.inf, "separated": True, "rank": f - extra}
        if fF is not None:                      # exact coupling supplied by the caller
            fF, exb, mode = np.asarray(fF, float), 0.0, "exact"
        else:
            fF, exb = self.coupling(F, s, t, x=x, mode=mode, r=r)
        YF = self.Y[:, self.pos[F]].astype(float)
        Dh = np.sqrt(alpha)
        Z = YF * Dh[None, :]
        if np.any(alpha < 1):
            Z = np.vstack([np.diag(np.sqrt(1 - alpha)), Z])
        z = Dh * fF
        rank = f - extra
        if rank == 0:
            return {"delta": 0.0, "separated": False, "rank": 0, "excess_bound": exb}
        U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
        sv, Vt = sv[:rank], Vt[:rank]
        coef = Vt @ z
        delta = float(np.sum((coef / sv) ** 2))
        if debias and mode == "exact" and np.all(alpha >= 1):
            delta *= (self.d - rank - 1) / self.d
        return {"delta": delta, "separated": False, "rank": rank, "excess_bound": exb,
                "smin2": float(sv[-1] ** 2)}


# ----------------------------------------------------------------------------
# Exact references
# ----------------------------------------------------------------------------


def exact_potential(solver, G, s, t, tol=1e-12):
    b = np.zeros(G.n); b[s] += 1; b[t] -= 1
    x, _ = solver.solve(b, tol=tol)
    return x


def exact_delta_woodbury(G, solver, F, s, t, alpha=None, x=None, cols=None):
    """Exact (up to solver accuracy) response by Woodbury on the base graph,
    using the cycle form C_FF = I - P_FF.  Singular scenarios are handled
    through the topological support (pseudo-inverse with exact rank)."""
    F = np.asarray(F, dtype=np.int64); f = len(F)
    alpha = np.ones(f) if alpha is None else np.asarray(alpha, float)
    if x is None:
        x = exact_potential(solver, G, s, t)
    if cols is None:
        Bf = np.zeros((G.n, f))
        Bf[G.u[F], np.arange(f)] += G.sw[F]
        Bf[G.v[F], np.arange(f)] -= G.sw[F]
        cols, _ = solver.solve(Bf, tol=1e-12)
    PFF = G.sw[F][:, None] * (cols[G.u[F]] - cols[G.v[F]])
    PFF = 0.5 * (PFF + PFF.T)
    CFF = np.eye(f) - PFF
    D = np.diag(alpha)
    M = np.eye(f) - D + np.sqrt(D) @ CFF @ np.sqrt(D)
    z = np.sqrt(alpha) * G.sw[F] * (x[G.u[F]] - x[G.v[F]])
    return M, z, x
