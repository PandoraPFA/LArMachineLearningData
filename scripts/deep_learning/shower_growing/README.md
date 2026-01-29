## Deep Learning 2D Shower Growing

Training and evaluation code for shower growing of 2D clusters with a [set transformer](https://arxiv.org/abs/1810.00825).

### Getting Setup

#### Python Environment

I used a python `3.13.1` venv with the following installs:
```
pip install torch numpy joblib leidenalg matplotlib networkx psutil igraph scikit-learn tqdm tqdm_joblib uproot pyyaml scipy
```

#### Exporting Training Data

Use the `LArDLTwoDShowerGrowing` in training mode over some simulation, preferably with roll-up turned off. I would recommend a sample of ~150k events. Hadd all the outputted training data ROOT files together.

Use the script `data/read_cluster.py` to process the ROOT file into training data. Some notes on the flags:
- I found the best hit feature vector preset to be 8, `--hit_feature_preset 8`
- To enable training with augmentations from iterative application of the prediction & clustering, you must have `--save_mc_cnts`
- For a mixed view training dataset, point `--out_dir_U`, `--out_dir_V`, and `--out_dir_W` all to the same directory.

The output will be a directory full of `.pt` files. Shuffle these into a directory structure like
```
my_dataset/
  train/
  val/
  test/
```

### Training Models

#### Running a Training

Trainings are configured through experiment yamls, see `experiments/example_experiment_config.yml`. They are submitted via the `train.py` script. The training will populate a directory in `checkpoints/` with loss values, validation examples, weight files etc.

#### Evaluating a Training

`test_sim.py` can be used to evaluate the raw similarity predictions, `test_clustering.py` can be used to evaluate the clustering from the similarity prediction. `test_clustering.py` expects a separate holdout root file made by hadding the ouput of `LArDLTwoDShowerGrowing` ran in training mode. The results of the evaluation will be written to the checkpoint directory of the experiment.

#### Exporting to Torchscript

`export_to_torchscript.py` does what it says with the option to validate the export (this takes a long time). The flag `--use_chunked_similarity_forward` is recommended, this performs the similarity MLP inference is serial chunks, which is the correct thing to do for CPU inference. Memory usage can be very high if this is not set.

### Notes on Code

```
clustering.py # Implementations of different clustering algorithms
config_parser.py # Parses the experiment yaml into a namedtuple and prepares the checkpoint dir
dataset.py # PyTorch hatasets and collation function
export_to_torchscript.py # Export weights from an experiment to torchscript
helpers.py # Assorted helpers: plotting, feature scaling, augmenting, ...
model.py # Main model class and model block classes
test_clustering.py # Evaluate the clustering based on a model's prediciton
test_sim.py # Evaluate a models similarty prediction
train.py # Script to start training
checkpoints/
  plot_lc.py # Plot the losses written by an experiment
data/
  event.py # Classes to represent hits and clusters
  read_cluster.py # Script to generate training data from ROOT file
slurm/ # Submission scripts for slurm
```

- `clustering.py` was originally for testing multiple clustering algorithms, it now just holds the implementation of the best algorithm, `connected_accessory_clustering`
- To reuse the Slurm submission scripts, you will need to edit the `#SBATCH` commands
