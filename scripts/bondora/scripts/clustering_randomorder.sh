#!/bin/bash
######
# Options

PROCESS=$1 # Start at 1
FILES_PER_PROCESS=$2
SETTINGS_FILE=$3
PNDR_SETUP_FILE=$4
SCRATCH_DIR=$5
DATA_DIR=$6
GEOMETRY_FILE=$7
INTERFACE_EXE=$8

######

i_start=$((${FILES_PER_PROCESS} * (${PROCESS} - 1)))
i_end=$((${i_start} + ${FILES_PER_PROCESS}))
# shuf reproducibly to avoid having only files from one flavour in a process, good for high n_files_per_process
input_files=$(echo -n `ls -1 ${DATA_DIR}/*.pndr | shuf --random-source=<(yes 1) | head -n ${i_end} | tail -n ${FILES_PER_PROCESS}` | sed 's/ /\:/g')

source $PNDR_SETUP_FILE
scratch_dir="${SCRATCH_DIR}/${PROCESS}"
mkdir -p $scratch_dir
cd $scratch_dir
rm -f ClusteringValidation.root

$INTERFACE_EXE -r AllHitsNu -i $SETTINGS_FILE -e $input_files -g $GEOMETRY_FILE

mv ClusteringValidation.root ../ClusteringValidation_${PROCESS}.root
cd ../
rm -r $scratch_dir
