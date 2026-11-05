#!/usr/bin/env bash
# Runs after run_timing_all.sh on the same dedicated core: Amazon with CHOLMOD (needs ~15 GB peak).
# The Jacobi-PCG run of run_timing_all.sh is kept as efficiency_Amazon_d256_jacobi.json.
set -u
PY=${PY:-python3}; CORE=${CORE:-2}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd "$(dirname "$0")/.."
L=results/logs
until [ -f $L/timing_all.done ]; do sleep 60; done
mv results/efficiency_Amazon_d256.json results/efficiency_Amazon_d256_jacobi.json
mv $L/eff_Amazon.log $L/eff_Amazon_jacobi.log
taskset -c $CORE timeout 10800 $PY experiments/exp_efficiency.py Amazon --d 256 --solver cholmod --resolve_q 1 > $L/eff_Amazon.log 2>&1
echo AMAZON_DONE > $L/timing_amazon.done
