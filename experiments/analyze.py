"""Aggregate result JSON files into tables (LaTeX snippets) and figures."""
import os, sys, json, math, glob, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
TAB = os.path.join(ROOT, "paper", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "figure.dpi": 150,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
STYLE = {  # method -> (label, color, marker)
    "ours-exact": ("CycleSketch", "#1b7837", "o"),
    "ours-exact-debiased": ("CycleSketch (debiased)", "#5aae61", "s"),
    "cut": ("CutSketch", "#c51b7d", "^"),
    "raw-cycle(0.0001)": ("NoCorr (tol 1e-4)", "#e08214", "v"),
    "fixedpoly(b=1e-2)": ("FixedPoly", "#2166ac", "D"),
    "wood-iter(1e-6)": ("Woodbury-iter (1e-6)", "#4d4d4d", "x"),
    "resolve-pcg(1e-4)": ("Resolve (1e-4)", "#8c510a", "+"),
    "ours-tree": ("CycleSketch-tree", "#a6dba0", "<"),
}
ORDER = ["PEG9241", "ACT25k", "ACT70k", "HepPh", "Enron", "Brightkite", "Gowalla", "RoadPA", "RoadTX", "RoadCA"]
MU_BINS = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]
MU_LAB = [r"$<10^{-4}$", r"$[10^{-4},10^{-3})$", r"$[10^{-3},10^{-2})$", r"$[10^{-2},10^{-1})$", r"$\geq10^{-1}$"]


def relerr(r, theta=1e-3):
    if math.isinf(r["delta"]):
        return None
    est = r["est"]
    if est is None or not np.isfinite(est) or est < 0:
        return np.inf
    return abs(est - r["delta"]) / max(r["delta"], theta * r["R0"])


def load_acc():
    data = {}
    for f in glob.glob(os.path.join(RES, "accuracy_*.json")):
        d = json.load(open(f))
        data[d["dataset"]] = d
    return data


def acc_table(data, d=256):
    """Median / p90 relative error (floor theta=1e-3) for significant responses,
    split by failure margin, per dataset and method."""
    methods = ["ours-exact", "ours-exact-debiased", "cut", "raw-cycle(0.0001)", "fixedpoly(b=1e-2)", "wood-iter(1e-6)", "resolve-pcg(1e-4)"]
    rows = []
    summary = {}
    for ds in ORDER:
        if ds not in data:
            continue
        recs = data[ds]["records"]
        for strat, cond in [("small", lambda mu: mu < 1e-2), ("large", lambda mu: mu >= 1e-2)]:
            for mth in methods:
                dd = 0 if mth in ("wood-iter(1e-6)", "resolve-pcg(1e-4)") else d
                errs = [relerr(r) for r in recs if r["method"] == mth and r["d"] == dd and cond(r["mu"])
                        and not math.isinf(r["delta"]) and r["delta"] >= 1e-3 * r["R0"]]
                errs = [e for e in errs if e is not None]
                if errs:
                    a = np.array(errs)
                    fail = float(np.mean(~np.isfinite(a)))
                    af = np.where(np.isfinite(a), a, 1e9)
                    fin = a[np.isfinite(a)]
                    summary[(ds, strat, mth)] = (float(np.median(af)), float(np.quantile(af, 0.9)), len(a), fail,
                                                 float(np.median(fin)) if len(fin) else float('nan'))
    return summary


def fig_accuracy(data, d=256):
    groups = [("Power grids", ["PEG9241", "ACT25k", "ACT70k"]), ("Road networks", ["RoadPA", "RoadTX", "RoadCA"]),
              ("Social networks", ["HepPh", "Enron", "Brightkite", "Gowalla"])]
    methods = ["ours-exact", "cut", "fixedpoly(b=1e-2)", "raw-cycle(0.0001)", "wood-iter(1e-6)", "resolve-pcg(1e-4)"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    for ax, (title, dss) in zip(axes, groups):
        present = [x for x in dss if x in data]
        for mth in methods:
            xs, ys, lo, hi = [], [], [], []
            for bi, (a, b) in enumerate(MU_BINS):
                errs = []
                for ds in present:
                    dd = 0 if mth in ("wood-iter(1e-6)", "resolve-pcg(1e-4)") else d
                    for r in data[ds]["records"]:
                        if r["method"] == mth and r["d"] == dd and a <= r["mu"] < b and not math.isinf(r["delta"]) \
                                and r["delta"] >= 1e-3 * r["R0"]:
                            e = relerr(r)
                            errs.append(1e3 if (e is None or not np.isfinite(e)) else min(e, 1e3))
                if len(errs) >= 5:
                    xs.append(bi); ys.append(np.median(errs)); lo.append(np.quantile(errs, 0.25)); hi.append(np.quantile(errs, 0.75))
            if xs:
                lab, col, mk = STYLE[mth]
                ax.plot(xs, ys, marker=mk, color=col, label=lab, lw=1.1, ms=3.5)
                ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
        if not present:
            ax.text(0.5, 0.5, "PLACEHOLDER\nresults not yet measured", transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color="#d95f02", bbox=dict(boxstyle="round", fc="#fff3e0", ec="#d95f02"))
        ax.set_yscale("log"); ax.set_title(f"{title}\n({', '.join(present)})" if 0 < len(present) else title, fontsize=7)
        ax.set_xticks(range(len(MU_BINS))); ax.set_xticklabels(MU_LAB, rotation=30, ha="right", fontsize=6)
        ax.set_xlabel(r"$\mu_F$ (smaller = closer to disconnection)")
        ax.set_ylim(1e-10, 2e3); ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("relative error of $\\Delta$")
    h, l = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        hh, ll = ax.get_legend_handles_labels()
        for a_, b_ in zip(hh, ll):
            if b_ not in l:
                h.append(a_); l.append(b_)
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.10), frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "accuracy_vs_margin.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
def dataset_table():
    st = json.load(open(os.path.join(RES, "dataset_stats.json")))
    fam = {"PEG9241": "power", "ACT25k": "power", "ACT70k": "power", "HepPh": "social", "Enron": "social",
           "Brightkite": "social", "Epinions": "social", "Gowalla": "social", "Amazon": "social",
           "RoadPA": "road", "RoadTX": "road", "RoadCA": "road"}
    order = ["PEG9241", "ACT25k", "ACT70k", "HepPh", "Enron", "Brightkite", "Epinions", "Gowalla", "Amazon", "RoadPA", "RoadTX", "RoadCA"]
    lines = []
    for ds in order:
        r = st[ds]
        tree = "tree_mst" if r["weighted"] else "tree_bfs"
        tau = r[tree]["tau"]; deg = r[tree]["deg_poly_beta1e-2"]
        if "cholmod" in r and "solve_s_per_rhs" in r["cholmod"] and ds != "Amazon":
            solver = "Chol."; ts = r["cholmod"]["solve_s_per_rhs"]
        else:
            solver = "Jac.-PCG"; ts = r["jacobi"]["solve_s_per_rhs"]
        def fm(x):
            if x >= 1e6: return f"{x/1e6:.2f}M"
            if x >= 1e3: return f"{x/1e3:.1f}K"
            return f"{x:.0f}"
        lines.append(f"{ds} & {fam[ds]} & {fm(r['n'])} & {fm(r['m'])} & {fm(r['r_cyc'])} & ${tau:.1e}$ & {deg:,} & {solver} & {ts*1e3:.1f} \\\\")
    body = "\n".join(lines).replace("e+0", "e").replace("e+", "e")
    import re
    body = re.sub(r"\$(\d\.\d)e(\d+)\$", lambda m_: f"${m_.group(1)}\\cdot10^{{{int(m_.group(2))}}}$", body)
    open(os.path.join(TAB, "datasets.tex"), "w").write(body + "\n")
    return body


def pivot_table():
    rows = []
    for ds in ["PEG9241", "ACT70k", "HepPh", "Enron"]:
        f = os.path.join(RES, f"pivot_{ds}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for r in d["per_d"]:
            c = r["coverage"]
            rows.append(f"{ds} & {r['d']} & {c['0.9']:.3f} & {c['0.95']:.3f} & {c['0.99']:.3f} & "
                        f"{r['mean_ratio_raw']:.3f} & {r['mean_ratio_debiased']:.3f} & {r['rel_std_debiased']:.3f} & {r['predicted_rel_std']:.3f} \\\\")
    open(os.path.join(TAB, "pivot.tex"), "w").write("\n".join(rows) + "\n")
    return rows


def fig_certificate(ds="ACT25k"):
    f = os.path.join(RES, f"certificate_{ds}.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    cf = [c for c in d["configs"] if c["solver"] != "fixedpoly"]
    tol = [c["tol"] for c in cf]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.7))
    ax = axes[0]
    ax.loglog(tol, [c["xi"] for c in cf], "o-", color="#1b7837", ms=3, label=r"certificate $\xi$")
    if "xi_frob" in cf[0]:
        ax.loglog(tol, [c["xi_frob"] for c in cf], "o--", color="#a6dba0", ms=3, label=r"Frobenius $\xi$")
    ax.loglog(tol, [max(c["dev_lift_max"], 1e-12) for c in cf], "s-", color="#2166ac", ms=3, label="observed (lift)")
    ax.set_xlabel("solver tolerance"); ax.set_ylabel("relative deviation")
    ax.legend(fontsize=5.5, frameon=False); ax.grid(alpha=0.3, lw=0.4); ax.invert_xaxis()
    ax = axes[1]
    ax.loglog(tol, [c["relerr_diag_lift_lowCee_med"] for c in cf], "o-", color="#1b7837", ms=3, label="with correction")
    ax.loglog(tol, [c["relerr_diag_raw_lowCee_med"] for c in cf], "v-", color="#e08214", ms=3, label="without correction")
    ax.axhline(cf[-1]["relerr_diag_ref_lowCee_med"], color="gray", ls=":", lw=0.8, label="exact probes")
    ax.set_xlabel("solver tolerance"); ax.set_ylabel(r"rel. err. of $\widehat C_{ee}$")
    ax.legend(fontsize=5.5, frameon=False); ax.grid(alpha=0.3, lw=0.4); ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"certificate_{ds}.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_coupling(dss=("ACT25k", "Enron")):
    present = [ds for ds in dss if os.path.exists(os.path.join(RES, f"coupling_{ds}.json"))]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(3.45, 1.75), squeeze=False)
    for ax, ds in zip(axes[0], present):
        d = json.load(open(os.path.join(RES, f"coupling_{ds}.json")))
        R = d["records"]
        js = sorted({r["j"] for r in R if r["mode"] != "tree"})
        for mode, col, mk, lab in [("corrected", "#1b7837", "o", "corrected"), ("uncorrected", "#c51b7d", "^", "uncorrected")]:
            for strat, ls, cond in [("small", "-", lambda mu: mu < 1e-2), ("large", ":", lambda mu: mu >= 1e-2)]:
                ys = []
                for j in js:
                    w = [(r["hi"] - r["lo"]) / max(r["delta"], 1e-3 * r["R0"]) for r in R if r["mode"] == mode and r["j"] == j and cond(r["mu"])]
                    ys.append(np.median(w) if w else np.nan)
                if np.all(np.isnan(ys)):
                    continue
                ax.loglog(js, ys, marker=mk, ls=ls, color=col, ms=3, lw=1,
                          label=f"{lab}, " + (r"$\mu<10^{-2}$" if strat == "small" else r"$\mu\geq10^{-2}$"))
        ax.set_title(ds); ax.set_xlabel("solver iterations $j$"); ax.grid(alpha=0.3, lw=0.4)
    axes[0][0].set_ylabel("certified interval width / $\\Delta$")
    axes[0][0].legend(fontsize=5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "coupling.pdf"), bbox_inches="tight"); plt.close(fig)


def coupling_summary():
    out = {}
    for ds in ["ACT25k", "Enron", "RoadPA"]:
        f = os.path.join(RES, f"coupling_{ds}.json")
        if not os.path.exists(f):
            continue
        R = json.load(open(f))["records"]
        for mode in ["corrected", "uncorrected", "tree"]:
            for j in sorted({r["j"] for r in R}):
                sel = [r for r in R if r["mode"] == mode and r["j"] == j]
                if not sel:
                    continue
                for strat, cond in [("small", lambda mu: mu < 1e-2), ("large", lambda mu: mu >= 1e-2)]:
                    ss = [r for r in sel if cond(r["mu"])]
                    if not ss:
                        continue
                    err = [abs(r["est"] - r["delta"]) / max(r["delta"], 1e-3 * r["R0"]) for r in ss]
                    wid = [(r["hi"] - r["lo"]) / max(r["delta"], 1e-3 * r["R0"]) for r in ss]
                    cov = [r["lo"] <= r["delta"] * (1 + 1e-12) and r["delta"] <= r["hi"] * (1 + 1e-12) for r in ss]
                    out[(ds, mode, j, strat)] = (float(np.median(err)), float(np.median(wid)), float(np.mean(cov)), len(ss),
                                                 float(np.median([r["eta"] for r in ss])))
    return out


def fmt_t(sec):
    """seconds -> compact string"""
    if sec < 0.1: return f"{sec*1e3:.0f}ms"
    if sec < 1: return f"{sec:.2f}s"
    if sec < 100: return f"{sec:.1f}s"
    if sec < 600: return f"{sec/60:.1f}m"
    if sec < 3600: return f"{sec/60:.0f}m"
    if sec < 86400 * 2: return f"{sec/3600:.1f}h"
    return f"{sec/86400:.1f}d"


def fmt_mb(b):
    mb = b / 1e6
    return f"{mb:.1f}" if mb < 10 else f"{mb:,.0f}"


def efficiency_table():
    st = json.load(open(os.path.join(RES, "dataset_stats.json")))
    order = ["PEG9241", "ACT25k", "ACT70k", "HepPh", "Enron", "Brightkite", "Epinions", "Gowalla", "Amazon", "RoadPA", "RoadTX", "RoadCA"]
    fsz = os.path.join(RES, "factor_sizes.json")      # factor sizes in the timing environment
    fsz = json.load(open(fsz)) if os.path.exists(fsz) else {}
    rows, raw = [], {}
    for ds in order:
        f = os.path.join(RES, f"efficiency_{ds}_d256.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f)); raw[ds] = r
        fac = fsz.get(ds, {}).get("factor_nnz") or st[ds].get("cholmod", {}).get("factor_nnz")
        fac_s = fmt_mb(fac * 12) if (fac and r["solver"] == "cholmod") else "--"
        q = r["query_exact_ms"]; qk2 = q.get("2", q.get(2))
        wood = r["woodbury_online_ms"]
        rows.append(f"{ds} & {fmt_t(r['t_build_total'])} & {fmt_mb(r['index_bytes'])} & {fac_s} & {qk2:.3f} & "
                    f"{wood:,.1f} & {wood/qk2:,.0f}$\\times$ & {fmt_t(r['scan_all_edges_s'])} & "
                    f"{fmt_t(r['lodf_cache_all_edges_extrapolated_s'])}$^\\dagger$ & {fmt_t(r['fixedpoly_build_extrapolated_s_beta1e-2'])}$^\\dagger$ \\\\")
    open(os.path.join(TAB, "efficiency.tex"), "w").write("\n".join(rows) + "\n")
    return rows, raw


def accuracy_latex(d=256):
    data = load_acc()
    s = acc_table(data, d)
    order = ["PEG9241", "ACT25k", "ACT70k", "RoadPA", "RoadTX", "RoadCA", "HepPh", "Enron", "Brightkite", "Gowalla"]
    def cell(key, finite_only=False):
        if key not in s:
            return "--"
        med, p90, nn, fail, medf = s[key]
        if finite_only:
            return f"{100*medf:.1f}" if np.isfinite(medf) else "--"

        if med >= 1e8:
            return "fail"
        p = "fail" if p90 >= 1e8 else f"{100*p90:.0f}"
        return f"{100*med:.1f} ({p})"
    def failc(key):
        return f"{100*s[key][3]:.0f}\\%" if key in s else "--"
    lines = ["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
             " & \\multicolumn{4}{c}{S: $\\mu_F<10^{-2}$} & \\multicolumn{4}{c}{L: $\\mu_F\\ge10^{-2}$}\\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}",
             "Dataset & $n$ & Ours & Cut & fail & $n$ & Ours & Cut & fail\\\\\\midrule"]
    def med(key, finite=False):
        if key not in s:
            return "--"
        m_, p90, nn, fail, medf = s[key]
        v = medf if finite else m_
        return "--" if (not np.isfinite(v) or v >= 1e8) else f"{100*v:.1f}"
    for ds in order:
        if ds not in data:
            continue
        nS = s.get((ds, "small", "ours-exact"), (0, 0, 0, 0, 0))[2]
        nL = s.get((ds, "large", "ours-exact"), (0, 0, 0, 0, 0))[2]
        lines.append(f"{ds} & {nS} & {med((ds,'small','ours-exact'))} & {med((ds,'small','cut'), True)} & {failc((ds,'small','cut'))} & "
                     f"{nL} & {med((ds,'large','ours-exact'))} & {med((ds,'large','cut'), True)} & {failc((ds,'large','cut'))}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "accuracy.tex"), "w").write("\n".join(lines) + "\n")
    return s


def fig_scalability():
    order = ["PEG9241", "ACT25k", "ACT70k", "HepPh", "Enron", "Brightkite", "Epinions", "Gowalla", "Amazon", "RoadPA", "RoadTX", "RoadCA"]
    fam_col = {"power": "#2166ac", "social": "#c51b7d", "road": "#1b7837"}
    fam = {"PEG9241": "power", "ACT25k": "power", "ACT70k": "power", "HepPh": "social", "Enron": "social", "Brightkite": "social",
           "Epinions": "social", "Gowalla": "social", "Amazon": "social", "RoadPA": "road", "RoadTX": "road", "RoadCA": "road"}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9))
    for ds in order:
        f = os.path.join(RES, f"efficiency_{ds}_d256.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        c = fam_col[fam[ds]]
        axes[0].loglog(r["m"], r["t_build_total"], "o", color=c, ms=4)
        axes[0].annotate(ds, (r["m"], r["t_build_total"]), fontsize=5, xytext=(2, 2), textcoords="offset points")
        q = r["query_exact_ms"]
        axes[1].loglog(r["m"], q["2"], "o", color=c, ms=4)
        axes[1].loglog(r["m"], r["woodbury_online_ms"], "x", color=c, ms=4)
    axes[0].set_xlabel("$m$"); axes[0].set_ylabel("index build (s)"); axes[0].grid(alpha=0.3, lw=0.4)
    axes[1].set_xlabel("$m$"); axes[1].set_ylabel("query latency (ms), $k=2$"); axes[1].grid(alpha=0.3, lw=0.4)
    from matplotlib.lines import Line2D
    axes[1].legend([Line2D([], [], marker="o", ls="", color="k"), Line2D([], [], marker="x", ls="", color="k")],
                   ["CycleSketch", "Woodbury (CHOLMOD)"], fontsize=5.5, frameon=False)
    ds_d = []
    for d in [32, 64, 128, 256, 512]:
        f = os.path.join(RES, f"efficiency_RoadPA_d{d}.json")
        if os.path.exists(f):
            r = json.load(open(f)); ds_d.append((d, r["t_build_total"], r["index_bytes"] / 1e9, r["query_exact_ms"]["2"]))
    if ds_d:
        ds_d = np.array(ds_d)
        ax = axes[2]
        ax.loglog(ds_d[:, 0], ds_d[:, 1], "o-", color="#1b7837", ms=3, label="build (s)")
        ax.loglog(ds_d[:, 0], ds_d[:, 2], "s-", color="#2166ac", ms=3, label="index (GB)")
        ax.loglog(ds_d[:, 0], ds_d[:, 3], "^-", color="#c51b7d", ms=3, label="query (ms)")
        ax.loglog(ds_d[:, 0], 100 * np.sqrt(2 / ds_d[:, 0]), "k:", lw=0.8, label=r"$100\sqrt{2/d}$ (% std)")
        ax.set_xlabel("sketch size $d$ (RoadPA)"); ax.legend(fontsize=5, frameon=False); ax.grid(alpha=0.3, lw=0.4)
    for c_, lab in [("#2166ac", "power"), ("#c51b7d", "social"), ("#1b7837", "road")]:
        axes[0].plot([], [], "o", color=c_, label=lab)
    axes[0].legend(fontsize=5.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "scalability.pdf"), bbox_inches="tight"); plt.close(fig)


def screening_table(order=("PEG9241", "ACT70k", "Enron", "RoadPA"), all_tau=("PEG9241",)):
    """Rows of Table 'screening' (paper format). Undecided share and #reported are means over the
    monitored pairs; precision/recall and top-100 recall are the worst case over pairs."""
    lines, info = [], {}
    for ds in order:
        f = os.path.join(RES, f"screening_{ds}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        taus = sorted({r["tau"] for r in d["runs"]})
        if ds not in all_tau:
            taus = [min(taus)]
        for sc in ["N-1", "N-2"]:
            for tau in taus:
                rr = [r for r in d["runs"] if r["scenario"] == sc and r["tau"] == tau]
                if not rr:
                    continue
                nsc = rr[0]["n_scen"]
                und = float(np.mean([r["n_undecided"] / r["n_scen"] for r in rr]))
                rep = float(np.mean([r["n_reported"] for r in rr]))
                if rr[0].get("precision") is not None:
                    # empty-set convention: precision (recall) is 1 when nothing is reported (true);
                    # older runs stored 0 for an empty reported AND empty true set
                    pr_ = [1.0 if r["n_reported"] == 0 else r["precision"] for r in rr]
                    rc_ = [1.0 if r["n_truth"] == 0 else r["recall"] for r in rr]
                    qual = f"{min(pr_):.2f}/{min(rc_):.2f}"
                elif rr[0].get("sample_agreement") is not None:
                    qual = f"{min(r['sample_agreement'] for r in rr):.4f}$^\\ddagger$"
                else:
                    qual = "--"
                if sc == "N-1" and "top100_recall_ours" in rr[0]:
                    top = f"{min(r['top100_recall_ours'] for r in rr):.2f} / " + (
                        f"{min(r['top100_recall_cut'] for r in rr):.2f}" if "top100_recall_cut" in rr[0] else "--")
                else:
                    top = "--"
                lines.append(f"{ds} & {sc} & {tau:g} & {nsc:,} & {100*und:.1f}\\% & {rep:,.0f} & {qual} & {top}\\\\")
                info[(ds, sc, tau)] = {"undecided_frac": und, "reported": rep, "n_pairs": len(rr),
                                       "t_filter_refine_s": [r["t_filter"] + r["t_refine"] for r in rr],
                                       "t_exhaustive": [r.get("t_exhaustive") for r in rr],
                                       "extrapolated": [r.get("exhaustive_extrapolated_s", r.get("exhaustive_cols_extrapolated_s")) for r in rr],
                                       "top100_ours": [r.get("top100_recall_ours") for r in rr],
                                       "top100_cut": [r.get("top100_recall_cut") for r in rr],
                                       "point_ours": [r.get("point_ours_prec_rec") for r in rr],
                                       "point_cut": [r.get("point_cut_prec_rec") for r in rr],
                                       "prec": [r.get("precision") for r in rr], "rec": [r.get("recall") for r in rr]}
    open(os.path.join(TAB, "screening.tex"), "w").write("\n".join(lines) + "\n")
    return lines, info


def fig_certificate_combined():
    from matplotlib.ticker import LogLocator, NullFormatter
    fig, axes = plt.subplots(1, 3, figsize=(4.6, 1.75))
    for ax, ds, solver in [(axes[0], "ACT25k", "AMG-PCG"), (axes[1], "Enron", "Jacobi-PCG")]:
        f = os.path.join(RES, f"certificate_{ds}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        cf = [c for c in d["configs"] if c["solver"] != "fixedpoly"]
        tol = [c["tol"] for c in cf]
        ax.loglog(tol, [c["xi"] for c in cf], "o-", color="#1b7837", ms=2.5, lw=1, label=r"bound $\xi$ (spectral)")
        ax.loglog(tol, [c["xi_frob"] for c in cf], "o--", color="#a6dba0", ms=2.5, lw=1, label=r"bound $\xi$ (Frobenius)")
        ax.loglog(tol, [max(c["dev_lift_max"], 1e-12) for c in cf], "s-", color="#2166ac", ms=2.5, lw=1, label="observed")
        ax.set_title(f"{ds} ({solver})", fontsize=6.5); ax.set_xlabel("tolerance", fontsize=6.5)
        ax.invert_xaxis(); ax.grid(alpha=0.3, lw=0.4)
        ax.yaxis.set_major_locator(LogLocator(numticks=5)); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=5.5)
    axes[0].set_ylabel("deviation and bound", fontsize=6.5)
    axes[0].legend(fontsize=5, frameon=False, loc="lower left")
    ax = axes[2]
    d = json.load(open(os.path.join(RES, "certificate_ACT25k.json")))
    cf = [c for c in d["configs"] if c["solver"] != "fixedpoly"]
    tol = [c["tol"] for c in cf]
    ax.loglog(tol, [100 * c["relerr_diag_lift_lowCee_med"] for c in cf], "o-", color="#1b7837", ms=2.5, lw=1, label="with correction")
    ax.loglog(tol, [100 * c["relerr_diag_raw_lowCee_med"] for c in cf], "v-", color="#e08214", ms=2.5, lw=1, label="without correction")
    fp = [c for c in d["configs"] if c["solver"] == "fixedpoly"]
    for c in fp:
        ax.plot([1e-9], [100 * c["relerr_diag_raw_lowCee_med"]], "v", color="#2166ac", ms=3)
        ax.plot([1e-9], [100 * c["relerr_diag_lift_lowCee_med"]], "o", color="#2166ac", ms=3)
    ax.plot([], [], "o", color="#2166ac", ms=3, label="FixedPoly, with corr.")
    ax.plot([], [], "v", color="#2166ac", ms=3, label="FixedPoly, w/o corr.")
    ax.axhline(100 * cf[-1]["relerr_diag_ref_lowCee_med"], color="gray", ls=":", lw=0.8)
    ax.set_title("ACT25k: error of $\\widehat C_{ee}$ (%)", fontsize=6.5); ax.set_xlabel("tolerance", fontsize=6.5)
    ax.invert_xaxis(); ax.grid(alpha=0.3, lw=0.4); ax.legend(fontsize=5, frameon=False)
    ax.yaxis.set_minor_formatter(NullFormatter()); ax.tick_params(labelsize=5.5)
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "certificate.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_coupling2(dss=("ACT25k", "RoadPA")):
    present = [ds for ds in dss if os.path.exists(os.path.join(RES, f"coupling_{ds}.json"))]
    fig, axes = plt.subplots(1, len(present), figsize=(2.3 * len(present), 1.75), squeeze=False)
    for ax, ds in zip(axes[0], present):
        R = json.load(open(os.path.join(RES, f"coupling_{ds}.json")))["records"]
        js = sorted({r["j"] for r in R if r["mode"] != "tree"})
        for mode, col, mk in [("corrected", "#1b7837", "o"), ("uncorrected", "#c51b7d", "^")]:
            for strat, ls, cond in [("small", "-", lambda mu: mu < 1e-2), ("large", ":", lambda mu: mu >= 1e-2)]:
                ys = []
                for j in js:
                    w = [(r["hi"] - r["lo"]) / max(r["delta"], 1e-3 * r["R0"]) for r in R if r["mode"] == mode and r["j"] == j and cond(r["mu"])]
                    ys.append(np.median(w) if len(w) >= 3 else np.nan)
                if np.all(np.isnan(ys)):
                    continue
                ax.loglog(js, ys, marker=mk, ls=ls, color=col, ms=2.5, lw=1,
                          label=("sketch-corrected" if mode == "corrected" else "uncorrected") + ", " + (r"$\mu_F<10^{-2}$" if strat == "small" else r"$\mu_F\geq10^{-2}$"))
        ax.set_title(ds, fontsize=6.5); ax.set_xlabel("solver iterations $j$", fontsize=6.5); ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(labelsize=5.5)
    axes[0][0].set_ylabel("interval width / $\\Delta$", fontsize=6.5)
    axes[0][0].legend(fontsize=4.8, frameon=False)
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "coupling.pdf"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    # Regenerate the tables/figures that depend on the result files; --all also rebuilds the
    # calibration table and the certificate/coupling figures (unchanged inputs -> same output).
    data = load_acc()
    print("accuracy datasets:", sorted(data))
    accuracy_latex(); fig_accuracy(data)
    efficiency_table(); screening_table()
    if "--all" in sys.argv:
        pivot_table(); fig_certificate_combined(); fig_coupling2(("ACT25k", "Enron"))
