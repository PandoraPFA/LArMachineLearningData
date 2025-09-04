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

  int trackRandIndexTotalDenom{0};
  double trackRandIndexTotal{0.};
  int showerPurityTotalDenom{0};
  double showerPurityTotal{0.};

  double trackRandIndex;
  int nTrackTrueClusters;
  int nTrackRecoClusters;
  double showerPurity;

  Event->SetBranchAddress("track_adjusted_rand_idx", &trackRandIndex);
  Event->SetBranchAddress("track_n_true_clusters", &nTrackTrueClusters);
  Event->SetBranchAddress("track_n_ari_reco_clusters", &nTrackRecoClusters);
  Event->SetBranchAddress("shower_purity", &showerPurity);

  for (int i = 0; i < Event->GetEntries(); i++)
  {
    Event->GetEntry(i);
    if (nTrackTrueClusters > 1 && nTrackRecoClusters > 1)
    {
      trackRandIndexTotal += trackRandIndex;
      trackRandIndexTotalDenom++;
    }
    if (showerPurity >= 0)
    {
      showerPurityTotal += showerPurity;
      showerPurityTotalDenom++;
    }
  }

  std::cout << trackRandIndexTotal / static_cast<double>(trackRandIndexTotalDenom) << ","
            << showerPurityTotal / static_cast<double>(showerPurityTotalDenom) << std::endl;' | sed "/^$/d" > result.txt # sed command removes empty line
