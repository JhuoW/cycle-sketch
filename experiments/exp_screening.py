"""EXP-G: certified N-1 / N-2 screening (filter-and-refine).

Threshold query: report every scenario F (|F|<=k) whose resistance increase
for a monitored pair satisfies Delta_F >= tau * R.
  * N-1: all edges of the graph.
  * N-2: all pairs inside a catalog of the M edges with the largest |f_e|.
Filter: exact per-query Wishart intervals (Theorem 3) with a Bonferroni split
of the failure probability over all scenarios and pairs; topology (bridges,
2-edge cuts) from the GF(2) labels.  Refine: exact Woodbury for undecided
scenarios.  Verification: exhaustive exact computation when feasible
(otherwise a random sample + extrapolated exhaustive cost).
"""
import sys, os, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from scipy.stats import chi2
from graphio import load
from core import Graph, SpanningTree, Solver, CycleSketch
from baselines import edge_cols

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--d", type=int, default=256)
ap.add_argument("--solver", default="cholmod")
ap.add_argument("--pairs", type=int, default=3)
ap.add_argument("--tau", type=float, nargs="+", default=[0.01, 0.1])
ap.add_argument("--M", type=int, default=1000)
ap.add_argument("--delta", type=float, default=0.01)
ap.add_argument("--exhaustive", action="store_true")
ap.add_argument("--verify_sample", type=int, default=2000)
ap.add_argument("--compare_cut", action="store_true")
ap.add_argument("--exact_cache", default=None,
                help="precomputed exact C_ee for all edges (experiments/exact_cdiag.py); used only for verification")
args = ap.parse_args()
out = {"dataset": args.dataset, "d": args.d, "M": args.M, "delta": args.delta, "runs": []}

n, u, v, w = load(args.dataset)
G = Graph(n, u, v, w, args.dataset)
weighted = bool(np.ptp(G.w) > 0)
t0 = time.time()
T = SpanningTree(G, kind="mst" if weighted else "bfs")
S = Solver(G, T.root, args.solver)
sk = CycleSketch(G, T, S, d=args.d, seed=11, store_dtype=np.float64)
t_build = time.time() - t0
out["t_build"] = t_build
print(f"[{args.dataset}] build {t_build:.1f}s", flush=True)
lab = sk.topo.lab
cutZ = None
if args.compare_cut:
    from baselines import CutGram
    cutZ = CutGram(G, S, d=args.d, seed=11).Z      # same probes as the CycleSketch
is_bridge = lab == 0
rng = np.random.default_rng(3)
d = args.d
t_col = None


def rank1_quad(a, b, c, f1, f2):
    """z^T M^+ z for 2x2 PSD [[a,b],[b,c]] of (topological) rank one, using
    its top eigenpair."""
    lam = (a + c) / 2 + np.sqrt(((a - c) / 2) ** 2 + b ** 2)
    v1, v2 = b, lam - a
    alt = np.abs(v1) + np.abs(v2) < 1e-300
    v1 = np.where(alt, 1.0, v1); v2 = np.where(alt, 0.0, v2)
    nrm = np.sqrt(v1 ** 2 + v2 ** 2)
    proj = (v1 * f1 + v2 * f2) / nrm
    return proj ** 2 / lam


def exact_Cdiag(E):
    """Exact C_ee via one solve per edge (blocked)."""
    out_ = np.empty(len(E))
    for b0 in range(0, len(E), 512):
        blk = E[b0:b0 + 512]
        X, _ = edge_cols(G, S, blk, tol=1e-12)
        out_[b0:b0 + len(blk)] = 1 - G.sw[blk] * (X[G.u[blk], np.arange(len(blk))] - X[G.v[blk], np.arange(len(blk))])
    return out_


# per-solve cost for extrapolation
t0 = time.time(); edge_cols(G, S, np.arange(min(64, G.m))); t_col = (time.time() - t0) / min(64, G.m)
out["t_solve_per_column"] = t_col

CACHE = {}
pairs = []
while len(pairs) < args.pairs:
    s, t = (int(a) for a in rng.choice(G.n, 2, replace=False))
    pairs.append((s, t))

n_tests_1 = G.m * len(pairs) * len(args.tau)
n_tests_2 = (args.M * (args.M - 1) // 2) * len(pairs) * len(args.tau)
dl1 = args.delta / 2 / n_tests_1
dl2 = args.delta / 2 / n_tests_2
q1lo, q1hi = chi2.ppf(dl1 / 2, d) / d, chi2.ppf(1 - dl1 / 2, d) / d          # rank 1
q2lo, q2hi = chi2.ppf(dl2 / 2, d - 1) / d, chi2.ppf(1 - dl2 / 2, d - 1) / d  # rank 2
q2lo1, q2hi1 = chi2.ppf(dl2 / 2, d) / d, chi2.ppf(1 - dl2 / 2, d) / d        # rank-1 pairs (2-cuts)
out["interval_factors"] = {"N1": [q1lo, q1hi], "N2": [q2lo, q2hi]}

for (s, t) in pairs:
    b = np.zeros(G.n); b[s] = 1; b[t] = -1
    tp = time.time()
    x, _ = S.solve(b, tol=1e-12)
    R = float(x[s] - x[t])
    f = G.sw * (x[G.u] - x[G.v])
    t_pair = time.time() - tp
    # ---------------- N-1 ----------------------------------------------------
    t1 = time.time()
    on_path = T.on_path(np.arange(G.m), s, t)
    Chat = np.einsum("ij,ij->j", sk.Y, sk.Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        Dhat = np.where(is_bridge, 0.0, f ** 2 / Chat)
    sep1 = is_bridge & on_path
    t_filter1 = time.time() - t1
    for tau in args.tau:
        thr = tau * R
        lo = Dhat * q1lo; hi = Dhat * q1hi
        pos = sep1 | (~is_bridge & (lo >= thr))
        neg = (~sep1) & (is_bridge | (hi < thr))
        und = ~(pos | neg)
        t2 = time.time()
        Eund = np.flatnonzero(und)
        Cex = exact_Cdiag(Eund) if len(Eund) else np.array([])
        ref_pos = Eund[f[Eund] ** 2 / Cex >= thr] if len(Eund) else np.array([], int)
        t_refine1 = time.time() - t2
        result = set(np.flatnonzero(pos).tolist()) | set(ref_pos.tolist())
        rec = {"pair": [s, t], "R": R, "tau": tau, "scenario": "N-1", "n_scen": G.m,
               "n_pos_cert": int(pos.sum()), "n_neg_cert": int(neg.sum()), "n_undecided": int(und.sum()),
               "n_reported": len(result), "t_pair_solve": t_pair, "t_filter": t_filter1, "t_refine": t_refine1,
               "exhaustive_extrapolated_s": G.m * t_col}
        # verification
        if args.exhaustive:
            nb = np.flatnonzero(~is_bridge)
            if "Call" not in CACHE:
                if args.exact_cache and os.path.exists(args.exact_cache):
                    # exact values computed offline with parallel workers (same solver, tol 1e-12);
                    # the single-threaded exhaustive cost is then extrapolated, not measured
                    CACHE["Call"] = np.load(args.exact_cache)[nb]
                    CACHE["t_exh1"] = None
                else:
                    te = time.time()
                    CACHE["Call"] = exact_Cdiag(nb)
                    CACHE["t_exh1"] = time.time() - te
            Call = CACHE["Call"]
            truth = set(nb[f[nb] ** 2 / Call >= thr].tolist()) | set(np.flatnonzero(sep1).tolist())
            if CACHE["t_exh1"] is not None:
                rec["t_exhaustive"] = CACHE["t_exh1"]
            else:
                rec["exhaustive_source"] = "parallel exact cache"
            rec["precision"] = len(result & truth) / len(result) if result else 1.0
            rec["recall"] = len(result & truth) / len(truth) if truth else 1.0
            rec["n_truth"] = len(truth)
            # decision quality of point estimates WITHOUT refinement: ours vs CutSketch
            Dtrue = np.zeros(G.m); Dtrue[nb] = f[nb] ** 2 / Call
            Dtrue[sep1] = np.inf
            def pr(est):
                pos_ = set(np.flatnonzero(est >= thr).tolist())
                return (len(pos_ & truth) / max(1, len(pos_)), len(pos_ & truth) / max(1, len(truth)))
            rec["point_ours_prec_rec"] = pr(np.where(sep1, np.inf, Dhat))
            K = 100
            topT = set(np.argsort(-Dtrue)[:K].tolist())
            rec["top100_recall_ours"] = len(set(np.argsort(-np.where(sep1, np.inf, Dhat))[:K].tolist()) & topT) / K
            if cutZ is not None:
                cee_cut = 1 - np.einsum("ij,ij->j", cutZ, cutZ)
                with np.errstate(divide="ignore", invalid="ignore"):
                    Dcut = np.where(is_bridge, 0.0, np.where(cee_cut > 0, f ** 2 / cee_cut, np.inf))
                Dcut = np.where(sep1, np.inf, Dcut)
                rec["point_cut_prec_rec"] = pr(Dcut)
                rec["top100_recall_cut"] = len(set(np.argsort(-Dcut)[:K].tolist()) & topT) / K
                rec["cut_infeasible_frac"] = float(np.mean((cee_cut <= 0) & ~is_bridge))
        else:
            samp = rng.choice(np.flatnonzero(~is_bridge), min(args.verify_sample, int((~is_bridge).sum())), replace=False)
            Cs = exact_Cdiag(samp)
            tr_s = f[samp] ** 2 / Cs >= thr
            rep_s = np.isin(samp, list(result))
            rec["sample_agreement"] = float(np.mean(tr_s == rep_s)); rec["sample_size"] = int(len(samp))
            rec["sample_true_pos"] = int(tr_s.sum())
        out["runs"].append(rec)
        print(json.dumps(rec), flush=True)
    # ---------------- N-2 within catalog ---------------------------------------
    cand = np.flatnonzero(~is_bridge)
    cat = cand[np.argsort(-np.abs(f[cand]))[: args.M]]
    t1 = time.time()
    Yc = sk.Y[:, cat]
    Gm = Yc.T @ Yc                      # M x M sketched cycle Gram
    fc = f[cat]
    iu, ju = np.triu_indices(len(cat), 1)
    a, c, bb = Gm[iu, iu], Gm[ju, ju], Gm[iu, ju]
    det = a * c - bb * bb
    two_cut = lab[cat][iu] == lab[cat][ju]          # {e,g} is a 2-edge cut (GF(2) labels equal)
    with np.errstate(divide="ignore", invalid="ignore"):
        Dh2 = (c * fc[iu] ** 2 - 2 * bb * fc[iu] * fc[ju] + a * fc[ju] ** 2) / det
        Dh2_cut = rank1_quad(a, bb, c, fc[iu], fc[ju])
    # separation for 2-cuts: s,t separated iff exactly one of the two... parity of path membership
    onp = T.on_path(cat, s, t)
    sep2 = two_cut & (onp[iu] ^ onp[ju])
    Dh2 = np.where(two_cut, Dh2_cut, Dh2)
    t_filter2 = time.time() - t1
    for tau in args.tau:
        thr = tau * R
        lo = np.where(two_cut, Dh2 * q2lo1, Dh2 * q2lo)
        hi = np.where(two_cut, Dh2 * q2hi1, Dh2 * q2hi)
        pos = sep2 | (~sep2 & (lo >= thr))
        neg = ~sep2 & (hi < thr)
        und = ~(pos | neg)
        t2 = time.time()
        # refine: exact columns for involved edges, exact 2x2 for undecided pairs
        U = np.flatnonzero(und)
        involved = np.unique(np.concatenate([cat[iu[U]], cat[ju[U]]])) if len(U) else np.array([], int)
        ref_pos = 0
        if len(U):
            idx = -np.ones(G.m, np.int64); idx[involved] = np.arange(len(involved))
            X, _ = edge_cols(G, S, involved, tol=1e-12)
            Pinv = G.sw[involved][:, None] * (X[G.u[involved]] - X[G.v[involved]])
            Pinv = 0.5 * (Pinv + Pinv.T)
            Cin = np.eye(len(involved)) - Pinv
            ei, ej = idx[cat[iu[U]]], idx[cat[ju[U]]]
            A_, C_, B_ = Cin[ei, ei], Cin[ej, ej], Cin[ei, ej]
            f1, f2 = fc[iu[U]], fc[ju[U]]
            ex = (C_ * f1 ** 2 - 2 * B_ * f1 * f2 + A_ * f2 ** 2) / (A_ * C_ - B_ ** 2)
            ex = np.where(two_cut[U], rank1_quad(A_, B_, C_, f1, f2), ex)
            refined_pos = U[ex >= thr]
            ref_pos = len(refined_pos)
        else:
            refined_pos = np.array([], int)
        t_refine2 = time.time() - t2
        reported = set(np.flatnonzero(pos).tolist()) | set(refined_pos.tolist())
        rec = {"pair": [s, t], "R": R, "tau": tau, "scenario": "N-2", "n_scen": int(len(iu)),
               "n_pos_cert": int(pos.sum()), "n_neg_cert": int(neg.sum()), "n_undecided": int(und.sum()),
               "n_refine_cols": int(len(involved)), "n_reported": len(reported), "t_filter": t_filter2, "t_refine": t_refine2,
               "exhaustive_cols_extrapolated_s": len(cat) * t_col}
        if args.exhaustive:
            te = time.time()
            X, _ = edge_cols(G, S, cat, tol=1e-12)
            Pc = G.sw[cat][:, None] * (X[G.u[cat]] - X[G.v[cat]]); Pc = 0.5 * (Pc + Pc.T)
            Cc = np.eye(len(cat)) - Pc
            A_, C_, B_ = Cc[iu, iu], Cc[ju, ju], Cc[iu, ju]
            ex = (C_ * fc[iu] ** 2 - 2 * B_ * fc[iu] * fc[ju] + A_ * fc[ju] ** 2) / (A_ * C_ - B_ ** 2)
            ex = np.where(two_cut, rank1_quad(A_, B_, C_, fc[iu], fc[ju]), ex)
            truth = set(np.flatnonzero(sep2 | (~sep2 & (ex >= thr))).tolist())
            rec["t_exhaustive"] = time.time() - te
            rec["precision"] = len(reported & truth) / len(reported) if reported else 1.0
            rec["recall"] = len(reported & truth) / len(truth) if truth else 1.0
            rec["n_truth"] = len(truth)
        out["runs"].append(rec)
        print(json.dumps(rec), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"screening_{args.dataset}.json"), "w"), indent=1, default=float)
