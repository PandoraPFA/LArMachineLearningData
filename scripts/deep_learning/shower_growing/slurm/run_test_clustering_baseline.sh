#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 1440
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --error=/springbrook/share/physics/phsajw/dl_cluster_merging/slurm_logs/err/job.%x.%j.err
#SBATCH --output=/springbrook/share/physics/phsajw/dl_cluster_merging/slurm_logs/out/job.%x.%j.out

CONFIG=$1
WEIGHTS=$2
TEST_FILE=$3
TEST_TREE=$4
VIEW=$5
FEATURE_PRESET=$6
CLUSTERING_MODE=$7
SUFFIX=$8
THRES=${9:-0.5}
BASELINE_FILE=${10}

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES
echo $CONFIG
echo $WEIGHTS
echo $TEST_FILE
echo $TEST_TREE
echo $VIEW
echo $FEATURE_PRESET
echo $CLUSTERING_MODE
echo $SUFFIX
echo $THRES
echo $BASELINE_FILE

cd /springbrook/share/physics/phsajw/dl_cluster_merging
source setup.sh
cd dl-cluster-assoc

python test_clustering.py --batch_mode \
                          --view $VIEW \
                          --hit_feature_preset $FEATURE_PRESET \
                          --clustering_mode $CLUSTERING_MODE \
                          --test_dir_suffix $SUFFIX \
                          --sim_threshold $THRES \
                          --baseline_root_file $BASELINE_FILE \
                          $CONFIG \
                          $WEIGHTS \
                          $TEST_FILE \
                          $TEST_TREE
