#!/bin/bash

export LOGDIR=${PWD}
export CACHE_DIR=/fs1/ECMWF/3dcorrection/data
export SLURM_NNODES=1
export SLURM_GPUS_ON_NODE=1
export PARAMS_PATH=${CACHE_DIR}/train

python trainer.py fit -c configs/cnn.yaml