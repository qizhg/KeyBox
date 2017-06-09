#!/bin/bash
export OMP_NUM_THREADS=4

th main.lua \
--batch_size 512 \
--nworker 4 \
--exp 'exp_PosSplit1_acting'
