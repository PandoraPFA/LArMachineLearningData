#!/bin/bash
######
# Options

PNDR_SETUP_FILE=$1
SCRATCH_DIR=$2

######

source $PNDR_SETUP_FILE
cd $SCRATCH_DIR

hadd ClusteringValidation_hadded.root *.root

root -b -l -q -n -e '
  TFile *f = TFile::Open("ClusteringValidation_hadded.root");
  TTree *Event = (TTree*)f->Get("Event");
  int nTotal{0};
  double randTotal{0.};
  double randIndex;
  int nTrueClusters;
  int nRecoClusters;
  Event->SetBranchAddress("adjusted_rand_idx", &randIndex);
  Event->SetBranchAddress("n_true_clusters", &nTrueClusters);
  Event->SetBranchAddress("n_ari_reco_clusters", &nRecoClusters);
  for (int i = 0; i < Event->GetEntries(); i++)
  {
    Event->GetEntry(i);
    if (nTrueClusters <= 1 || nRecoClusters <= 1) { continue; }
    randTotal += randIndex;
    nTotal++;
  }
  std::cout << randTotal / nTotal << std::endl;' | sed "/^$/d" > result.txt # sed command removes empty line
