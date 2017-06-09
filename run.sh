#!/bin/bash
export OMP_NUM_THREADS=4

th main.lua \
--batch_size 128 \
--nworker 4 \
--exp 'exp_PosSplit_acting'
