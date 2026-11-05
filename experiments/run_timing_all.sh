#!/usr/bin/env bash
# All timing-sensitive runs, executed SEQUENTIALLY, single-threaded, pinned to one
# dedicated core (CPU 2) of a CCD that no other experiment job uses (CPUs 0-7, 32-39 kept free).
# Usage: PY=/path/to/python bash experiments/run_timing_all.sh
set -u
PY=${PY:-python3}
CORE=${CORE:-2}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd "$(dirname "$0")/.."
L=results/logs; mkdir -p $L
R="taskset -c $CORE timeout 7200 $PY"
for ds in PEG9241 ACT25k ACT70k HepPh Enron Brightkite Epinions RoadPA RoadTX RoadCA; do
  $R experiments/exp_efficiency.py $ds --d 256 --solver cholmod > $L/eff_$ds.log 2>&1
done
$R experiments/exp_efficiency.py Gowalla --d 256 --solver cholmod --resolve_q 1 > $L/eff_Gowalla.log 2>&1
$R experiments/exp_efficiency.py Amazon --d 256 --solver jacobi --resolve_q 0 --wood_q 5 --nq 600 > $L/eff_Amazon.log 2>&1
for d in 32 64 128 512; do
  $R experiments/exp_efficiency.py RoadPA --d $d --solver cholmod --resolve_q 0 --wood_q 5 > $L/eff_RoadPA_d$d.log 2>&1
done
$R experiments/exp_topology.py ACT70k --nq 3000 --bfs_q 3000 > $L/topo_ACT70k.log 2>&1
$R experiments/exp_topology.py Enron --nq 1000 --bfs_q 1000 > $L/topo_Enron.log 2>&1
$R experiments/exp_topology.py RoadPA --nq 3000 --bfs_q 3000 > $L/topo_RoadPA.log 2>&1
$R experiments/exp_topology.py RoadCA --nq 3000 --bfs_q 3000 > $L/topo_RoadCA.log 2>&1
$R experiments/exp_screening.py PEG9241 --d 256 --pairs 2 --M 300 --exhaustive --compare_cut > $L/screen_PEG9241.log 2>&1
$R experiments/exp_screening.py ACT70k --d 256 --pairs 3 --M 1000 --exhaustive --compare_cut > $L/screen_ACT70k.log 2>&1
$R experiments/exp_screening.py Enron --d 256 --pairs 3 --M 1000 --exhaustive --compare_cut > $L/screen_Enron.log 2>&1
# RoadPA: verification uses exact C_ee precomputed in parallel (experiments/exact_cdiag.py)
until [ -f results/cache/cdiag_RoadPA.npy ]; do sleep 60; done
$R experiments/exp_screening.py RoadPA --d 256 --pairs 3 --M 1000 --exhaustive --compare_cut \
   --exact_cache results/cache/cdiag_RoadPA.npy > $L/screen_RoadPA.log 2>&1
echo TIMING_ALL_DONE > $L/timing_all.done
