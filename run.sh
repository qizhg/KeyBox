#!/bin/bash
export OMP_NUM_THREADS=1

th main.lua \
--batch_size 128 \
--nworker 1 \
--exp_id 19
