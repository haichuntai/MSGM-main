#!/bin/bash

export LD_LIBRARY_PATH=/home/kpn/anaconda3/envs/open/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -u ../../tools/train.py ../../configs/dicl/prw_dicl.py

