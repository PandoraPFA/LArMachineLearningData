#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 1440
#SBATCH --gres=gpu:1
#SBATCH --error=/springbrook/share/physics/phsajw/dl_cluster_merging/slurm_logs/err/job.%x.%j.err
#SBATCH --output=/springbrook/share/physics/phsajw/dl_cluster_merging/slurm_logs/out/job.%x.%j.out

CONFIG=$1
WEIGHTS=$2
ADJ_THRES=${3:-0.5}
UVW=${4:-0}
ITER_AUGS=${5:-0}

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES
echo $CONFIG
echo $WEIGHTS
echo $ADJ_THRES
echo $UVW
echo $ITER_AUGS

cd /springbrook/share/physics/phsajw/dl_cluster_merging
source setup.sh
cd dl-cluster-assoc

uvw_arg=""
if [ "$UVW" -eq 1 ];
then
  uvw_arg="--UVW"
fi
iter_augs_arg=""
if [ "$ITER_AUGS" -eq 1 ];
then
  iter_augs_arg="--iterative_augs"
fi

python test_sim.py --batch_mode \
                   --adjacency_threshold $ADJ_THRES \
                   $uvw_arg \
                   $iter_augs_arg \
                   $CONFIG \
                   $WEIGHTS
