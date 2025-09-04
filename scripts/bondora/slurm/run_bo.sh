#!/bin/bash
#SBATCH -N 1
#SBATCH -c 48
#SBATCH -J bo
#SBATCH --exclusive
#SBATCH --mem 0
#SBATCH --exclude=taskfarm188,dedicated-cqv
#SBATCH --partition=compute,epp
#SBATCH --error=/storage/epp2/phsajw/optuna_bo/slurm_logs/err/%x.%j.err
#SBATCH --output=/storage/epp2/phsajw/optuna_bo/slurm_logs/out/%x.%j.out

######
# Options

SETTINGS_FILE=$1

WORK_DIR="/storage/epp2/phsajw/optuna_bo/optuna_bo"
VENV_SETUP="/storage/epp2/phsajw/optuna_bo/setup_venv.sh"

######

cd $WORK_DIR
source $VENV_SETUP
python bo_pndr.py $SETTINGS_FILE
