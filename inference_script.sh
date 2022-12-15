#!/bin/bash

#SBATCH --job-name yolox_visdrone
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=10G
#SBATCH --time 3-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -w sw9
#SBATCH -o slurm/logs/slurm-%A-%x.out


echo 'extracting the dataset from NAS ...'
#mkdir -p /local_datasets/VisDrone
#cp -r -u /data/datasets/VisDrone /local_datasets/VisDrone
echo 'done'

"""
echo 'setup yolox environment ...'
pip3 install -v -e .  # or  python3 setup.py develop
echo 'done'
"""

current_time=$(date +'%Y%m%d-%H%M%S')
echo $current_time

#python tools/train.py -f exps/network_slim/yolox_s_slim.py -d 1 -b 64 --fp16 -c pretrained_models/yolox_nano.pth
#python -m yolox.tools.eval -f exps/network_slim/yolox_s_slim.py -d 1 -b 64 --fp16 -c ./YOLOX_outputs/yolox_s_slim_sparsity_train/best_ckpt.pth --test
"""
python tools/demo.py image -f exps/network_slim/yolox_s_slim.py -c ./YOLOX_outputs/yolox_s_slim_sparsity_train/best_ckpt.pth --path /local_datasets/VisDrone/test2017/0000001_02999_d_0000005.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
python tools/demo.py image -f exps/network_slim/yolox_s_slim.py -c ./YOLOX_outputs/yolox_s_slim_sparsity_train/best_ckpt.pth --path /local_datasets/VisDrone/test2017/0000153_01601_d_0000001.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
python tools/demo.py image -f exps/network_slim/yolox_s_slim.py -c ./YOLOX_outputs/yolox_s_slim_sparsity_train/best_ckpt.pth --path /local_datasets/VisDrone/test2017/0000244_00001_d_0000001.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
python tools/demo.py image -f exps/network_slim/yolox_s_slim.py -c ./YOLOX_outputs/yolox_s_slim_sparsity_train/best_ckpt.pth --path /local_datasets/VisDrone/test2017/0000276_04401_d_0000529.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
python tools/demo.py image -f exps/network_slim/yolox_s_slim.py -c ./YOLOX_outputs/yolox_s_slim_sparsity_train/best_ckpt.pth --path /local_datasets/VisDrone/test2017/0000289_02401_d_0000823.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
python tools/demo.py image -f exps/network_slim/yolox_s_slim.py -c ./YOLOX_outputs/yolox_s_slim_sparsity_train/best_ckpt.pth --path /local_datasets/VisDrone/test2017/0000313_07001_d_0000471.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
python tools/demo.py image -f exps/network_slim/yolox_s_slim.py -c ./YOLOX_outputs/yolox_s_slim_sparsity_train/best_ckpt.pth --path /local_datasets/VisDrone/test2017/0000335_04117_d_0000064.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
"""

python3 tools/demo.py image \
-f exps/network_slim/yolox_s_slim.py \
-c ./YOLOX_outputs/yolox_s_slim_fine_tuning/latest_ckpt.pth \
--path /local_datasets/VisDrone/test2017/0000001_02999_d_0000005.jpg  --save_result

python3 tools/demo.py image \
-f exps/network_slim/yolox_s_slim.py \
-c ./YOLOX_outputs/yolox_s_slim_fine_tuning/latest_ckpt.pth \
--path /local_datasets/VisDrone/test2017/0000153_01601_d_0000001.jpg  --save_result

python3 tools/demo.py image \
-f exps/network_slim/yolox_s_slim.py \
-c ./YOLOX_outputs/yolox_s_slim_fine_tuning/latest_ckpt.pth \
--path /local_datasets/VisDrone/test2017/0000244_00001_d_0000001.jpg  --save_result

python3 tools/demo.py image \
-f exps/network_slim/yolox_s_slim.py \
-c ./YOLOX_outputs/yolox_s_slim_fine_tuning/latest_ckpt.pth \
--path /local_datasets/VisDrone/test2017/0000276_04401_d_0000529.jpg  --save_result

python3 tools/demo.py image \
-f exps/network_slim/yolox_s_slim.py \
-c ./YOLOX_outputs/yolox_s_slim_fine_tuning/latest_ckpt.pth \
--path /local_datasets/VisDrone/test2017/0000289_02401_d_0000823.jpg  --save_result

python3 tools/demo.py image \
-f exps/network_slim/yolox_s_slim.py \
-c ./YOLOX_outputs/yolox_s_slim_fine_tuning/latest_ckpt.pth \
--path /local_datasets/VisDrone/test2017/0000313_07001_d_0000471.jpg  --save_result

python3 tools/demo.py image \
-f exps/network_slim/yolox_s_slim.py \
-c ./YOLOX_outputs/yolox_s_slim_fine_tuning/latest_ckpt.pth \
--path /local_datasets/VisDrone/test2017/0000335_04117_d_0000064.jpg  --save_result
echo 'done'
# letting slurm know this code finished without any problem
exit 0