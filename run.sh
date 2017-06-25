#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

th main.lua \
--batch_size 10 \
--nworker 2 \
--exp 'exp_conv_ContComm'
