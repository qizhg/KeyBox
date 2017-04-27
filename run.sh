#!/bin/bash
export OMP_NUM_THREADS=4

th main.lua \
--batch_size 64 \
--nworker 4 \
--exp_id 26
