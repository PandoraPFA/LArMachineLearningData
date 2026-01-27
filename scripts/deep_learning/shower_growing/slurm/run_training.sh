#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 2880
#SBATCH --gres=gpu:1
#SBATCH --error=/springbrook/share/physics/phsajw/dl_cluster_merging/slurm_logs/err/job.%x.%j.err
#SBATCH --output=/springbrook/share/physics/phsajw/dl_cluster_merging/slurm_logs/out/job.%x.%j.out

CONFIG=$1

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES
echo $CONFIG

cd /springbrook/share/physics/phsajw/dl_cluster_merging
source setup.sh
cd dl-cluster-assoc

python train.py $CONFIG
