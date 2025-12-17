Hello!

This directory contains all notebooks used to train the DLThreeDClusterSplittingAlgorithm (https://github.com/imawby/LArContent/blob/feature/DLClusterSplittingRelease/larpandoradlcontent/LArTwoDReco/DLThreeDClusterSplittingAlgorithm.cc).

1. First create a 'files' and 'models' directory (within this directory). 

2. Run the DLThreeDClusterSplittingAlgorithm in training mode to obtain the training files. I usually setup my grid jobs such that each job runs on a portion of the full dataset, and therefore end up with N training files. I reserve at least one of these to use in the 'Performance' notebooks.

3. Each training file must first be passed through Windows.ipynb. In the initial training file, each entry in the training trees corresponds to one cluster. The feature time-series, which form the model's input, are therefore sequences of varying length. Windows.ipynb firstly removes any clusters which we do not want to train on, and then splits the feature time-series sequences into windows of a defined length. These windows, and corresponding truth labels, are saved into new .npy files to be used in training. If one needs to modify a feature i.e. normalise or cap the value of feature it should be done here.

4. EncoderTraining.ipynb is used to train (you guessed it) the encoder model. This is the model which predicts whether a given window is 0) a track and not contaminated i.e. contains hits from multiple MCParticles 1) a track and contaminated or 2) part of a true shower-like particle (electron/photon). This script will save a snapshot of the model after each epoch in the 'models' directory.

5. EncoderPerformance.ipynb is used to investigate the performance of the trained encoder model. Here one can look at the performance of any saved model in terms of classification distributions and create confusion matrices.

6. EncoderDecoderTraining.ipynb is used to train (wow, you're so good at this) the encoder-decoder model. This is a model which predicts whether each position in the window is 0) not a true split position or 1) a true split position. Again, a snapshot of the model after each epoch is saved in the 'models' directory. 

7. EncoderDecoderPerformance.ipynb is analagous to the notebook described in point 5.

There are a few more files that do the 'behind the scenes' work:

Datasets.py - Details the Encoder and EncoderDecoder datasets.
Models.py - Defines the Encoder and EncoderDecoder models.
TrainingMetrics.py - Stores the functions that create the classification distributions and confusion matrices.
Utilities.py - Stores the functions used to process the input data.

Good luck!
