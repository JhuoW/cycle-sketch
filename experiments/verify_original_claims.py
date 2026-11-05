"""Independent numerical verification of the mathematical claims in the
ERGraph3 exposition ("Conservation-Preserving Cycle Sketches").

Every check is done with dense linear algebra on small random graphs, so the
script is an independent reference implementation, not a reuse of the
original artifact.  Output: results/verify_original_claims.json
"""
import json, math, os, sys
import numpy as np

rng = np.random.default_rng(20261103)
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "verify_original_claims.json")


def random_connected_graph(n, extra, rng, wlo=0.1, whi=10.0):
    """Random spanning tree plus `extra` random edges (parallel edges allowed)."""
    edges = []
    perm = rng.permutation(n)
    for i in range(1, n):
        edges.append((perm[i], perm[rng.integers(0, i)]))
    for _ in range(extra):
        u, v = rng.choice(n, 2, replace=False)
        edges.append((u, v))
    w = np.exp(rng.uniform(np.log(wlo), np.log(whi), len(edges)))
    return np.array(edges), w


def grounded_incidence(n, edges, w, ground=0):
    m = len(edges)
    A = np.zeros((n, m))
    for j, (u, v) in enumerate(edges):
        A[u, j] += math.sqrt(w[j])
        A[v, j] -= math.sqrt(w[j])
    keep = [i for i in range(n) if i != ground]
    return A[keep, :], keep


def max_weight_tree(n, edges, w):
    """Kruskal on decreasing conductance; returns list of tree edge indices."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree = []
    for j in np.argsort(-w):
        u, v = edges[j]
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            tree.append(j)
    return sorted(tree)


def components_after(n, edges, removed):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    rem = set(removed)
    for j, (u, v) in enumerate(edges):
        if j in rem:
            continue
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    return [find(i) for i in range(n)]


def eff_res_dense(n, edges, w, s, t):
    """Effective resistance by pseudo-inverse of the full Laplacian on the
    component containing s (inf if t is elsewhere)."""
    comp = components_after(n, edges, [])
    if comp[s] != comp[t]:
        return math.inf
    nodes = [i for i in range(n) if comp[i] == comp[s]]
    idx = {v: k for k, v in enumerate(nodes)}
    Lc = np.zeros((len(nodes), len(nodes)))
    for j, (u, v) in enumerate(edges):
        if u in idx and v in idx:
            a, b = idx[u], idx[v]
            Lc[a, a] += w[j]; Lc[b, b] += w[j]; Lc[a, b] -= w[j]; Lc[b, a] -= w[j]
    b = np.zeros(len(nodes)); b[idx[s]] = 1; b[idx[t]] = -1
    return float(b @ np.linalg.pinv(Lc) @ b)


results = {}

# ---------------------------------------------------------------------------
# 1. Response formula with singular scenarios and the four-cycle example.
# ---------------------------------------------------------------------------
edges4 = np.array([(0, 1), (1, 2), (2, 3), (3, 0)])
w4 = np.ones(4)
A4, keep4 = grounded_incidence(4, edges4, w4, ground=0)
L4 = A4 @ A4.T
P4 = A4.T @ np.linalg.solve(L4, A4)
C4 = np.eye(4) - P4
F = [1, 3]
MF = C4[np.ix_(F, F)]
# paper's vertices 1..4 are 0..3 here; pair (1,2) -> (0,1); vertex 0 grounded,
# so b = e_0 - e_1 has grounded coordinates -e_1.
b12 = np.zeros(3); b12[keep4.index(1)] = -1.0
x = np.linalg.solve(L4, b12)
z = A4[:, F].T @ x
Wf = np.array([[1.0], [1.0]]) / math.sqrt(2)
delta_4cycle = ((Wf.T @ z) ** 2 / (Wf.T @ MF @ Wf)).item()
results["four_cycle"] = {
    "MF": MF.tolist(),
    "z": z.tolist(),
    "R_base": float(b12 @ x),
    "delta_pair_(1,2)": delta_4cycle,
    "expected_delta": 0.25,
    "ok": abs(delta_4cycle - 0.25) < 1e-12 and abs(float(b12 @ x) - 0.75) < 1e-12,
}

# Random singular / partial scenarios against direct pseudo-inverse references.
max_err = 0.0; n_fin = 0; n_inf = 0; n_checked = 0
for trial in range(60):
    n = int(rng.integers(6, 14)); extra = int(rng.integers(0, 8))
    edges, w = random_connected_graph(n, extra, rng)
    m = len(edges)
    A, keep = grounded_incidence(n, edges, w)
    L = A @ A.T
    Linv = np.linalg.inv(L)
    f = int(rng.integers(1, min(5, m) + 1))
    F = list(rng.choice(m, f, replace=False))
    alpha = np.where(rng.random(f) < 0.6, 1.0, rng.uniform(0.05, 0.95, f))
    D = np.diag(alpha)
    B = A[:, F] @ np.sqrt(D)
    M = np.eye(f) - B.T @ Linv @ B
    w2 = w.copy()
    for j, e in enumerate(F):
        w2[e] = (1 - alpha[j]) * w[e]
    removed = [F[j] for j in range(f) if alpha[j] == 1.0]
    comp = components_after(n, edges, removed)
    edges2 = np.array([edges[j] for j in range(m) if w2[j] > 0])
    w2p = np.array([w2[j] for j in range(m) if w2[j] > 0])
    for _ in range(6):
        s, t = rng.choice(n, 2, replace=False)
        bvec = np.zeros(n); bvec[s] = 1; bvec[t] = -1
        bg = bvec[keep]
        xx = Linv @ bg
        zz = B.T @ xx
        R0 = float(bg @ xx)
        Rref = eff_res_dense(n, edges2, w2p, s, t) if len(edges2) else math.inf
        # compatibility test through topology
        if comp[s] != comp[t]:
            n_inf += 1
            ok = math.isinf(Rref)
            max_err = max(max_err, 0.0 if ok else 1.0)
            continue
        n_fin += 1
        Mp = np.linalg.pinv(M, rcond=1e-12)
        dlt = float(zz @ Mp @ zz)
        rel = abs((R0 + dlt) - Rref) / max(Rref, 1e-300)
        max_err = max(max_err, rel)
        n_checked += 1
results["singular_response_formula"] = {"finite_cases": n_fin, "separated_cases": n_inf,
                                        "max_rel_err_vs_pinv_reference": max_err}

# ---------------------------------------------------------------------------
# 2./3. Repaired covariance identity and the exact distortion identity.
# ---------------------------------------------------------------------------
cov_err = 0.0; beta_err = 0.0; sandwich_ok = True
for trial in range(30):
    n = int(rng.integers(6, 16)); extra = int(rng.integers(1, 12))
    edges, w = random_connected_graph(n, extra, rng)
    m = len(edges)
    A, keep = grounded_incidence(n, edges, w)
    L = A @ A.T
    Linv = np.linalg.inv(L)
    P = A.T @ Linv @ A
    C = np.eye(m) - P
    tree = max_weight_tree(n, edges, w)
    AT = A[:, tree]
    T = np.zeros((m, n - 1)); T[tree, :] = np.linalg.inv(AT)
    assert np.allclose(A @ T, np.eye(n - 1))
    J = np.eye(m) - T @ A
    H = Linv + 0.3 * rng.standard_normal((n - 1, n - 1)) / n  # nonsymmetric, probe-independent
    Q = J @ (np.eye(m) - A.T @ H @ A)
    EQ = Q - C
    cov_err = max(cov_err, np.abs(Q @ Q.T - (C + EQ @ EQ.T)).max(), np.abs(EQ @ C).max(), np.abs(C @ EQ - EQ).max())
    beta = np.linalg.norm(EQ, 2) ** 2
    LT = AT @ AT.T
    evals, evecs = np.linalg.eigh(L)
    Lh = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    S = Lh @ np.linalg.inv(LT) @ Lh
    Rop = np.eye(n - 1) - Lh @ H @ Lh
    se, sv = np.linalg.eigh(S - np.eye(n - 1))
    Sh = sv @ np.diag(np.sqrt(np.maximum(se, 0))) @ sv.T
    beta2 = np.linalg.norm(Sh @ Rop, 2) ** 2
    beta_err = max(beta_err, abs(beta - beta2) / max(beta, 1e-300))
    # sandwich C <= QQ^T <= (1+beta) C on random vectors
    for _ in range(20):
        v = rng.standard_normal(m)
        lhs = v @ C @ v; mid = v @ Q @ Q.T @ v; rhs = (1 + beta) * lhs
        if not (lhs <= mid + 1e-9 and mid <= rhs + 1e-9):
            sandwich_ok = False
results["repaired_covariance"] = {"max_identity_residual": float(cov_err),
                                  "max_rel_err_beta_identity": float(beta_err),
                                  "sandwich_holds_on_random_vectors": sandwich_ok}

# ---------------------------------------------------------------------------
# 4. Third-kind Chebyshev polynomial, envelope, hyperbolic evaluation,
#    and the APPLY_FIXED_POLYNOMIAL pseudo-code.
# ---------------------------------------------------------------------------

def V(l, x):
    v0, v1 = np.ones_like(x), 2 * x - 1
    if l == 0:
        return v0
    for _ in range(l - 1):
        v0, v1 = v1, 2 * x * v1 - v0
    return v1


env_err = 0.0; hyp_err = 0.0; minimax_ok = True
for Lam in [1.5, 3.0, 10.0, 100.0, 1e4]:
    a = math.asinh(1 / math.sqrt(Lam - 1))
    for l in [0, 1, 2, 5, 9]:
        lam = np.linspace(1, Lam, 200001)
        xi = (2 * lam - Lam - 1) / (Lam - 1)
        xi0 = -(Lam + 1) / (Lam - 1)
        q = V(l, xi) / V(l, np.array(xi0))
        val = np.max((lam - 1) * q ** 2)
        env = 1 / math.sinh((2 * l + 1) * a) ** 2
        env_err = max(env_err, abs(val - env) / env)
        hyp = abs(float(V(l, np.array(-math.cosh(2 * a)))))
        hyp_err = max(hyp_err, abs(hyp - math.sinh((2 * l + 1) * a) / math.sinh(a)) / hyp)
        # first-kind comparison: (Lam-1) sech^2(2 l a)
        if l >= 1:
            Tl = np.cos(l * np.arccos(np.clip(xi, -1, 1)))
            T0 = math.cosh(l * math.acosh(-xi0)) * (-1) ** l
            val1 = np.max((lam - 1) * (Tl / T0) ** 2)
            if val1 < val * (1 - 1e-9):
                minimax_ok = False
results["third_kind_envelope"] = {"max_rel_err_envelope": env_err, "max_rel_err_hyperbolic_eval": hyp_err,
                                  "first_kind_never_better_on_grid": minimax_ok}


def apply_fixed_polynomial(v_rhs, L, TreeSolve, Lam, degree):
    """Literal transcription of the pseudo-code in Section 5.5."""
    if degree == 0:
        return np.zeros_like(v_rhs)
    a = 2 / (Lam - 1)
    c = -(Lam + 1) / (Lam - 1)
    nu = 2 * c - 1
    h_prev = np.zeros_like(v_rhs)
    h_cur = (-2 * a / nu) * TreeSolve(v_rhs)
    for j in range(1, degree):
        nu_next = 2 * c - 1 / nu
        res = v_rhs - L @ h_cur
        h_next = (2 * c / nu_next) * h_cur - h_prev / (nu * nu_next) - (2 * a / nu_next) * TreeSolve(res)
        h_prev, h_cur = h_cur, h_next
        nu = nu_next
    return h_cur


poly_err = 0.0
for trial in range(10):
    n = int(rng.integers(8, 16)); extra = int(rng.integers(2, 10))
    edges, w = random_connected_graph(n, extra, rng)
    m = len(edges)
    A, keep = grounded_incidence(n, edges, w)
    L = A @ A.T
    tree = max_weight_tree(n, edges, w)
    AT = A[:, tree]; LT = AT @ AT.T
    LTinv = np.linalg.inv(LT)
    tau = np.trace(L @ LTinv) - (n - 1)
    Lam = 1 + tau
    evals, evecs = np.linalg.eigh(L)
    Lh = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    S = Lh @ LTinv @ Lh
    sev, svec = np.linalg.eigh(S)
    for deg in [1, 2, 3, 6]:
        H = np.column_stack([apply_fixed_polynomial(np.eye(n - 1)[:, i], L, lambda r: LTinv @ r, Lam, deg)
                             for i in range(n - 1)])
        Rop = np.eye(n - 1) - Lh @ H @ Lh
        xi = (2 * sev - Lam - 1) / (Lam - 1)
        qS = svec @ np.diag(V(deg, xi) / V(deg, np.array(-(Lam + 1) / (Lam - 1)))) @ svec.T
        poly_err = max(poly_err, np.abs(Rop - qS).max())
results["apply_fixed_polynomial_pseudocode"] = {"max_abs_err_R_vs_q(S)": float(poly_err)}

# ---------------------------------------------------------------------------
# 5. Degree needed: third kind vs first kind for target beta*=1e-4.
# ---------------------------------------------------------------------------
deg_table = []
for Lam in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
    a = math.asinh(1 / math.sqrt(Lam - 1))
    bstar = 1e-4
    l3 = max(0, math.ceil(0.5 * (math.asinh(1 / math.sqrt(bstar)) / a - 1)))
    # first kind: (Lam-1) sech^2(2 l a) <= bstar
    l1 = math.ceil(math.acosh(math.sqrt((Lam - 1) / bstar)) / (2 * a))
    deg_table.append({"Lambda": Lam, "deg_third_kind": l3, "deg_first_kind": l1})
results["degree_table_beta_1e-4"] = deg_table

# ---------------------------------------------------------------------------
# 6. Total stretch (trace bound) on representative small graphs vs lambda_max.
# ---------------------------------------------------------------------------
stretch = []
for (nx_, ny_) in [(8, 8), (16, 16), (24, 24)]:
    n = nx_ * ny_
    edges = []
    for i in range(nx_):
        for j in range(ny_):
            v = i * ny_ + j
            if i + 1 < nx_: edges.append((v, v + ny_))
            if j + 1 < ny_: edges.append((v, v + 1))
    edges = np.array(edges); w = np.ones(len(edges)) * (1 + 1e-9 * rng.random(len(edges)))
    A, keep = grounded_incidence(n, edges, w)
    L = A @ A.T
    tree = max_weight_tree(n, edges, w)
    AT = A[:, tree]; LT = AT @ AT.T
    M_ = np.linalg.solve(LT, L)
    ev = np.linalg.eigvals(M_).real
    tau = float(np.trace(M_) - (n - 1))
    stretch.append({"grid": f"{nx_}x{ny_}", "n": n, "m": len(edges), "trace_bound_Lambda": 1 + tau,
                    "true_lambda_max_S": float(ev.max())})
results["trace_bound_vs_true_lambda_max"] = stretch

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(results, fh, indent=2)
print(json.dumps(results, indent=2))
