#!/usr/bin/env bash
# The first d=256 efficiency runs overlapped with parallel (non-timing) jobs on other CCDs, which
# share memory bandwidth.  Re-run all of them on the same dedicated core once nothing else runs;
# the overlapped measurements are kept in results/repro/overlapped/ for comparison only.
set -u
PY=${PY:-python3}; CORE=${CORE:-2}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd "$(dirname "$0")/.."
L=results/logs; mkdir -p results/repro/overlapped $L/overlapped
until [ -f $L/timing_amazon.done ]; do sleep 60; done
for ds in PEG9241 ACT25k ACT70k HepPh Enron Brightkite Epinions RoadPA RoadTX RoadCA Gowalla; do
  cp results/efficiency_${ds}_d256.json results/repro/overlapped/ && cp $L/eff_$ds.log $L/overlapped/
  extra=""; [ $ds = Gowalla ] && extra="--resolve_q 1"
  taskset -c $CORE timeout 7200 $PY experiments/exp_efficiency.py $ds --d 256 --solver cholmod $extra > $L/eff_$ds.log 2>&1
done
echo IDLE_DONE > $L/timing_idle.done
