#!/bin/bash

export PYTHONPATH="$(pwd):$(pwd)/speed_estimation/modules/depth_map/Unidepth"

python3.10 -m speed_estimation.speed_estimation "$@" 