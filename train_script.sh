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
cp -r -u /data/datasets/VisDrone /local_datasets/VisDrone
echo 'done'

"""
echo 'setup yolox environment ...'
pip3 install -v -e .  # or  python3 setup.py develop
echo 'done'
"""

current_time=$(date +'%Y%m%d-%H%M%S')
echo $current_time

python tools/train.py -f exps/example/custom/visdrone_nano.py -d 1 -b 64 --fp16 -c pretrained_models/yolox_nano.pth --cache


echo 'done'
# letting slurm know this code finished without any problem
exit 0