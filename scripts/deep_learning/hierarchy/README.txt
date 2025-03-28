Hello!

This directory contains all notebooks used to train the DLNeutrinoHierarchyAlgorithm (https://github.com/imawby/LArContent/blob/feature/MLPNeutrinoHierarchy/larpandoradlcontent/LArThreeDReco/LArEventBuilding/DLNeutrinoHierarchyAlgorithm.cc).

1. You must first create a 'files' and 'models' directory (within this directory). You will need to run the DLNeutrinoHierarchy alg in training mode to obtain our training file, put this file inside the 'files' directory.

2. The scripts to train the primary models are found in the PrimaryTier directory. One first needs to run the 'WritePrimaryTierFile.ipynb' script in both track and shower mode. 'TrainPrimaryTierModel_tracks.ipynb' and 'TrainPrimaryTierModel_showers.ipynb' can then be ran to create the primary tier models.

3. The scripts to train the later-tier models are found in the LaterTier directory. One first needs to run the 'WriteLaterTierFile.ipynb' script in both track and shower mode. 'TrainLaterTierModel' can then be ran in both track and shower mode to create the later tier models.

4. The scripts used to tune the DLNeutrinoHierarchy alg parameters can be found in the 'Metrics' directory. The order of running is 'AddScores.ipynb' -> 'TuneCut.ipynb' -> 'HierarchyMetrics.ipynb'.

For more details, see the 'Interaction hierarchy building' section of the Pandora_ML_Guide.pdf (https://github.com/imawby/Documentation/blob/feature/MLDocs/ML_Documentation/Pandora_ML_Guide.pdf).

Good luck!
