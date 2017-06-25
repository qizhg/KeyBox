#!/bin/bash
export OMP_NUM_THREADS=4

th main.lua \
--batch_size 10 \
--nworker 2 \
--exp 'exp_conv_monitoring'
