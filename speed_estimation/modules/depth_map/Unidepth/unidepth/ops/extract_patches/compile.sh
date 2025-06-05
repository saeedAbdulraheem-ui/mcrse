#!/usr/bin/env bash

if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
fi

export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH

python3.10 setup.py build install
