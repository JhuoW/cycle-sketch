#!/usr/bin/env bash
# Quiet re-run of the factorization-free (Jacobi-PCG) Amazon build; the first measurement overlapped
# with other jobs. Index-only measurements (no baselines). Runs after run_timing_idlecheck.sh.
set -u
PY=${PY:-python3}; CORE=${CORE:-2}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd "$(dirname "$0")/.."
L=results/logs
until [ -f $L/timing_idle.done ]; do sleep 60; done
taskset -c $CORE timeout 14400 $PY experiments/exp_efficiency.py Amazon --d 256 --solver jacobi --nq 600 --skip_baselines \
   --out results/efficiency_Amazon_d256_jacobi.json > $L/eff_Amazon_jacobi.log 2>&1
echo JAC_DONE > $L/timing_jacobi.done
