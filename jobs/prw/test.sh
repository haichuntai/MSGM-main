#!/bin/bash

export LD_LIBRARY_PATH=/home/kpn/anaconda3/envs/open/lib:$LD_LIBRARY_PATH

config_name='prw_dicl' 
num_epoch='25'
CUDA_VISIBLE_DEVICES=0 python ../../tools/test.py ../../configs/dicl/${config_name}.py ../../../DICL-mian/jobs/prw/work_dirs/prw_dicl/epoch_${num_epoch}.pth --eval bbox --out work_dirs/${config_name}/results_1000.pkl >work_dirs/${config_name}/log_tmp_${num_epoch}.txt
CUDA_VISIBLE_DEVICES=0 python ../../tools/test_results_prw.py ${config_name} >work_dirs/${config_name}/result_${config_name}_${num_epoch}.txt
