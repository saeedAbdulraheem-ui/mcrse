#!/bin/bash

export PYTHONPATH="$(pwd):$(pwd)/speed_estimation/modules/depth_map/Unidepth"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export TORCH_COMPILE_DISABLE=1
python3.10 -m speed_estimation.speed_estimation "$@" 