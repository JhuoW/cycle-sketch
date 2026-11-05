"""Values for the paper's running example (Fig. 1)."""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from core import Graph, SpanningTree, Solver, CycleSketch
from baselines import truth, woodbury_CFF, coupling_exact
# vertices v1..v8 -> 0..7
E = [(0,1),(1,2),(2,3),(3,0),(0,2),   # cluster A
     (4,5),(5,6),(6,7),(7,4),(5,7),   # cluster B
     (3,4),                           # e*  : link
     (1,6)]                           # e_o : weak bypass
w = [1]*10 + [1.0, 0.01]
E = np.array(E); G = Graph(8, E[:,0], E[:,1], np.array(w, float))
T = SpanningTree(G, kind="mst"); S = Solver(G, T.root, "cholmod")
Lp = np.linalg.pinv(G.L.toarray())
Phi = G.BW.toarray().T
C = np.eye(G.m) - Phi @ Lp @ Phi.T
s, t = 0, 6
b = np.zeros(8); b[s] = 1; b[t] = -1
R = b @ Lp @ b
x = Lp @ b; f = G.sw * (x[G.u] - x[G.v])
out = {"R_st": R, "leverage": {f"e{j}": 1 - C[j, j] for j in range(G.m)}, "Cdiag": {f"e{j}": C[j, j] for j in range(G.m)},
       "flow": {f"e{j}": f[j] for j in range(G.m)}}
scen = {"F={e*}": [10], "F={e*,e_o}": [10, 11], "F={(v1,v3)}": [4], "F={(v1,v2),(v4,v1)}": [0, 3], "F={e*,(v3,v4)}": [10, 2]}
for name, F in scen.items():
    tr = truth(G, S, F, s, t, x=x)
    out[name] = {k: v for k, v in tr.items()}
    if not math.isinf(tr["delta"]):
        CFF = C[np.ix_(F, F)]
        out[name]["CFF"] = CFF.tolist()
        out[name]["fF"] = f[F].tolist()
# cut-space vs cycle-space distortion of 5% on F={e*}
e = 10; Pi = 1 - C[e, e]
for zeta in [0.01, 0.05]:
    c_cut = 1 - (1 + zeta) * Pi
    c_cyc = (1 - zeta) * C[e, e]
    d_true = f[e] ** 2 / C[e, e]
    out[f"distort_{zeta}"] = {"C_true": C[e, e], "C_cut": c_cut, "C_cyc": c_cyc,
                              "Delta_true": d_true, "Delta_cut": f[e] ** 2 / c_cut if c_cut > 0 else float('inf'),
                              "Delta_cyc": f[e] ** 2 / c_cyc}
print(json.dumps(out, indent=1, default=float))
json.dump(out, open("results/running_example.json", "w"), indent=1, default=float)
