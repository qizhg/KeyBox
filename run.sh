#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

th main.lua \
--batch_size 1024 \
--nworker 4 \
--exp 'exp_PosSplit100_2a_1D_std00'
