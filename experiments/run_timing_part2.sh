#!/usr/bin/env bash
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd "$(dirname "$0")/.."
L=results/logs; mkdir -p $L
for ds in RoadPA RoadTX RoadCA; do
  timeout 3600 python3 experiments/exp_efficiency.py $ds --d 256 --solver cholmod > $L/eff_$ds.log 2>&1
done
for d in 32 64 128 512; do
  timeout 3600 python3 experiments/exp_efficiency.py RoadPA --d $d --solver cholmod --resolve_q 0 --wood_q 5 > $L/eff_RoadPA_d$d.log 2>&1
done
timeout 3600 python3 experiments/exp_topology.py RoadPA --nq 3000 --bfs_q 300 > $L/topo_RoadPA.log 2>&1
timeout 3600 python3 experiments/exp_topology.py RoadCA --nq 3000 --bfs_q 200 > $L/topo_RoadCA.log 2>&1
timeout 3600 python3 experiments/exp_efficiency.py Gowalla --d 256 --solver cholmod --resolve_q 1 > $L/eff_Gowalla.log 2>&1
timeout 5400 python3 experiments/exp_efficiency.py Amazon --d 256 --solver jacobi --resolve_q 0 --wood_q 5 --nq 600 > $L/eff_Amazon.log 2>&1
echo PART2_DONE > $L/timing2.done
