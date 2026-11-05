#!/usr/bin/env bash
# Accuracy runs (machine-independent numbers) for the remaining graphs, run concurrently on
# CPUs outside the timing CCD.  Usage: PY=/path/to/python bash experiments/run_accuracy_rest.sh
set -u
PY=${PY:-python3}
CPUS=${CPUS:-8-31,40-63}
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
cd "$(dirname "$0")/.."
L=results/logs; mkdir -p $L
R="taskset -c $CPUS $PY"
export WORKLOADS=endpoint,local,random
# reproducibility check of an already reported dataset in this environment (separate output file)
$R experiments/exp_accuracy.py PEG9241 --d 64 256 --nq 70 --fixedpoly --out results/repro/accuracy_PEG9241.json > $L/repro_acc_PEG9241.log 2>&1 &
$R experiments/exp_accuracy.py Gowalla --d 64 256 --nq 70 --iter_solver jacobi > $L/acc_Gowalla.log 2>&1 &
for ds in RoadPA RoadTX RoadCA; do
  $R experiments/exp_accuracy.py $ds --d 64 256 --nq 70 --iter_solver amg --skip_online --raw_d 64 > $L/acc_$ds.log 2>&1 &
done
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 $R experiments/exact_cdiag.py RoadPA --workers 20 > $L/cdiag_RoadPA.log 2>&1 &
wait
echo ACC_DONE > $L/acc_rest.done
