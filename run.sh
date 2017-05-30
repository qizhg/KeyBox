#!/bin/bash
export OMP_NUM_THREADS=2

th main.lua \
--batch_size 256 \
--nworker 2 \
--exp 'exp_2all_acting_RNN'
