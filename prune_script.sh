#!/bin/bash

#SBATCH --job-name yolox_visdrone
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=10G
#SBATCH --time 3-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -w sw9
#SBATCH -o slurm/logs/slurm-%A-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')
echo $current_time

#python tools/train.py -f exps/example/custom/visdrone_nano.py -d 1 -b 64 --fp16 -c pretrained_models/yolox_nano.pth --cache
#python3 prun_train/prune.py -c YOLOX_outputs/visdrone_nano/latest_ckpt.pth -f exps/example/custom/visdrone_nano.py -expn pruned_yolox_nano
echo '==== Generate pruning schema ===='

python3 tools/gen_pruning_schema.py --save-path exps/network_slim/yolox_s_schema.json --name yolox-s 

echo '==== Sparsity training ===='
python3 tools/train.py -d 1 -b 32 \
-f exps/network_slim/yolox_s_slim_train.py \
-expn yolox_s_slim_sparsity_train \
-c pretrained_models/yolox_s.pth \
--fp16 \
--cache \
network_slim_sparsity_train_s 0.0001

echo '==== Network Slimming ===='
python3 tools/train.py -d 1 -b 32 \
-f exps/network_slim/yolox_s_slim.py \
-c ./YOLOX_outputs/yolox_s_slim_sparsity_train/latest_ckpt.pth \
-expn yolox_s_slim_fine_tuning \
--fp16 \
--cache \
network_slim_ratio 0.65


echo 'done'
# letting slurm know this code finished without any problem
exit 0