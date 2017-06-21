#!/bin/bash
export OMP_NUM_THREADS=4

th main.lua \
--batch_size 10 \
--nworker 1 \
--exp 'exp_CornerPos_acting'
