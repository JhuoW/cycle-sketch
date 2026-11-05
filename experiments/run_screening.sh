#!/usr/bin/env bash
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd "$(dirname "$0")/.."
L=results/logs; mkdir -p $L
timeout 5400 python3 experiments/exp_screening.py ACT70k --d 256 --pairs 3 --M 1000 --exhaustive --compare_cut > $L/screen_ACT70k.log 2>&1
timeout 5400 python3 experiments/exp_screening.py Enron --d 256 --pairs 3 --M 1000 --exhaustive --compare_cut > $L/screen_Enron.log 2>&1
timeout 7200 python3 experiments/exp_screening.py RoadPA --d 256 --pairs 3 --M 1000 > $L/screen_RoadPA.log 2>&1
echo SCREEN_DONE > $L/screen.done
