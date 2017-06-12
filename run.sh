#!/bin/bash
export OMP_NUM_THREADS=4

th main.lua \
--batch_size 1024 \
--nworker 4 \
--exp 'exp_PosSplit_2a_1D_std00'
