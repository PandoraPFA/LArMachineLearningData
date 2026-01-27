#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 2880
#SBATCH --error=/springbrook/share/physics/phsajw/dl_cluster_merging/slurm_logs/err/job.%x.%j.err
#SBATCH --output=/springbrook/share/physics/phsajw/dl_cluster_merging/slurm_logs/out/job.%x.%j.out

CONFIG=$1
WEIGHTS=$2
VAL_ROOT_FILE=$3
VAL_TREE_NAME=$4
VAL_HIT_PRESET=$5

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES
echo $CONFIG
echo $WEIGHTS
echo $VAL_ROOT_FILE
echo $VAL_TREE_NAME
echo $VAL_HIT_PRESET

cd /springbrook/share/physics/phsajw/dl_cluster_merging
source setup.sh
cd dl-cluster-assoc

python export_torchscript.py --batch_mode \
                             --use_chunked_similarity_forward \
                             --validation_treename $VAL_TREE_NAME \
                             --validation_hit_preset $VAL_HIT_PRESET \
                             --validation_root_file $VAL_ROOT_FILE \
                             $CONFIG \
                             $WEIGHTS
