
config_name="cuhk_dicl"
config_path="../../configs/dicl/${config_name}.py"
num_epoch='25'
CUDA_VISIBLE_DEVICES=0 python -u ../../tools/test_personsearch.py $config_path  work_dirs/${config_name}/epoch_${num_epoch}.pth --eval bbox >> work_dirs/${config_name}/result_${config_name}_${num_epoch}.txt 2>&1


