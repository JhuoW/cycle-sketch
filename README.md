# CycleSketch — certified what-if effective-resistance queries under edge failures

Implementation of the CycleSketch index, the baselines, and every experiment, timing and analysis script.
Datasets (`data/`) and experiment outputs (`results/`) are not part of the repository: the scripts read the
downloaded datasets from `data/raw/` and write raw JSON results and logs to `results/`.

## Layout

| Path | Content |
|---|---|
| `src/graphio.py` | dataset loaders (SNAP via GraphChallenge mirror, MATPOWER), LCC preprocessing |
| `src/core.py` | graph, spanning tree + O(n) tree kernels (Numba), solvers (CHOLMOD / PyAMG / Jacobi-PCG), **CycleSketch** index (lift, spectral certificate ξ), GF(2) topology oracle, what-if queries |
| `src/baselines.py` | CutSketch, RawCycle, Woodbury (online), Resolve, FixedPoly (a-priori construction), exact ground truth |
| `experiments/verify_original_claims.py` | numerical verification of the algebraic identities the method relies on |
| `experiments/test_core.py` | unit tests against dense linear algebra |
| `experiments/exp_accuracy.py` | accuracy vs failure margin |
| `experiments/exp_partial.py` | partial failures (edge weakening) |
| `experiments/exp_certificate.py` | solver-agnostic error bound with inexact solvers |
| `experiments/exp_pivot.py` | calibration of the exact confidence intervals |
| `experiments/exp_coupling.py` | sketch-corrected pair flow from inexact potentials |
| `experiments/exp_efficiency.py` | efficiency and scalability (single-threaded) |
| `experiments/exp_topology.py` | XOR connectivity test vs BFS |
| `experiments/exp_screening.py` | certified N-1 / N-2 screening |
| `experiments/exact_cdiag.py` | exact cycle mass C_ee of every edge with parallel workers (ground truth only) |
| `experiments/factor_sizes.py` | CHOLMOD factor sizes in the timing environment |
| `experiments/run_timing_all.sh`, `run_timing_amazon_cholmod.sh`, `run_timing_amazon_jacobi.sh`, `run_timing_idlecheck.sh` | timing runs (sequential, single-threaded, pinned to one dedicated core) |
| `experiments/run_accuracy_rest.sh` | accuracy runs for Gowalla and the road networks, plus a reproducibility re-run of PEG9241 |
| `experiments/run_timing_part*.sh`, `run_screening.sh` | earlier scripts used on a 4-core VM (superseded) |
| `experiments/analyze.py` | aggregates the result files into LaTeX tables and PDF figures (output directories are set at the top of the script) |

## Reproduce

```bash
conda env create -f environment.yml -n cyclesketch && conda activate cyclesketch   # exact versions used
# (or: pip install numpy scipy pyamg networkx matplotlib numba scikit-sparse; needs libsuitesparse-dev)
python3 experiments/test_core.py            # expect "fails: 0"
python3 experiments/verify_original_claims.py
python3 src/graphio.py                      # downloads must already be in data/raw (see below)
python3 experiments/exp_accuracy.py PEG9241 --d 64 256 --nq 100 --fixedpoly
PY=$(which python) bash experiments/run_accuracy_rest.sh     # accuracy on Gowalla and road graphs
PY=$(which python) bash experiments/run_timing_all.sh        # idle core; single-threaded, pinned (taskset)
python3 experiments/analyze.py
```

Raw data: SNAP graphs from `https://graphchallenge.s3.amazonaws.com/snap/<name>/<name>_adj.tsv`
(roadNet-PA/TX/CA, email-Enron, ca-HepPh, loc-brightkite_edges, soc-Epinions1, loc-gowalla_edges, amazon0601)
and MATPOWER cases from `https://raw.githubusercontent.com/MATPOWER/matpower/master/data/<case>.m`
(case2869pegase, case9241pegase, case_ACTIVSg25k, case_ACTIVSg70k), saved as `data/raw/<name>.tsv|.m`.

## Notes

* **Timing protocol.** Run timings single-threaded (OPENBLAS/OMP/NUMBA/MKL threads = 1), pinned to one core
  (`taskset -c 2`), and with no other experiment running. Concurrent jobs on other cores slowed the efficiency
  runs by 2–110% through shared memory bandwidth. The `run_timing_*.sh` scripts follow this protocol.
* **Exact cycle masses.** On large graphs, one exact solve per edge takes hours, so `experiments/exact_cdiag.py`
  computes the exact C_ee = 1 − w_e R_e with parallel CHOLMOD workers and caches them in
  `results/cache/cdiag_<dataset>.npy`. Their sum equals m − n + 1 to machine precision.
  Set the BLAS/OMP thread counts to 1 so that the workers do not oversubscribe the cores.
* **Amazon without factorization.** `exp_efficiency.py Amazon --d 256 --solver jacobi --skip_baselines`
  builds the index with Jacobi-preconditioned CG (about 480 iterations per probe, see
  `run_timing_amazon_jacobi.sh`).
