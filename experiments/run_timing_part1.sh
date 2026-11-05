#!/usr/bin/env bash
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd "$(dirname "$0")/.."
L=results/logs; mkdir -p $L
for ds in PEG9241 ACT25k ACT70k HepPh Enron Brightkite Epinions; do
  timeout 3600 python3 experiments/exp_efficiency.py $ds --d 256 --solver cholmod > $L/eff_$ds.log 2>&1
done
echo PART1_DONE > $L/timing1.done
