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

  int randIndexTotalDenom{0};
  double randIndexTotal{0.};
  int trackPurityTotalDenom{0};
  double trackPurityTotal{0.};
  int trackCompletenessTotalDenom{0};
  double trackCompletenessTotal{0.};

  double randIndex;
  int nTrueClusters;
  int nRecoClusters;
  double trackPurity;
  double trackCompleteness;

  Event->SetBranchAddress("adjusted_rand_idx", &randIndex);
  Event->SetBranchAddress("n_true_clusters", &nTrueClusters);
  Event->SetBranchAddress("n_ari_reco_clusters", &nRecoClusters);
  Event->SetBranchAddress("track_purity", &trackPurity);
  Event->SetBranchAddress("track_completeness", &trackCompleteness);

  for (int i = 0; i < Event->GetEntries(); i++)
  {
    Event->GetEntry(i);
    if (nTrueClusters > 1 && nRecoClusters > 1)
    {
      randIndexTotal += randIndex;
      randIndexTotalDenom++;
    }
    if (trackPurity >= 0)
    {
      trackPurityTotal += trackPurity;
      trackPurityTotalDenom++;
    }
    if (trackCompleteness >= 0)
    {
      trackCompletenessTotal += trackCompleteness;
      trackCompletenessTotalDenom++;
    }
  }

  std::cout << randIndexTotal / static_cast<double>(randIndexTotalDenom) << ","
            << trackPurityTotal / static_cast<double>(trackPurityTotalDenom) << ","
            << trackCompletenessTotal / static_cast<double>(trackCompletenessTotalDenom) << std::endl;' | sed "/^$/d" > result.txt # sed command removes empty line
