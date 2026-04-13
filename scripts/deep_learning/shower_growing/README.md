## Deep Learning 2D Shower Growing

Training and evaluation code for shower growing of 2D clusters with a [set transformer](https://arxiv.org/abs/1810.00825).

This repository contains the end-to-end workflow for preparing training data, training shower growing models, evaluating their performance, and exporting them for inference.

The intended workflow is:
1. Export cluster training data as a ROOT file using Pandora (not part of this repo)
2. Convert it into `.pt` datasets with `data/read_cluster.py`
3. Train from an experiment yaml with `train.py`
4. Evaluate with `test_sim.py` and/or `test_clustering.py`
5. Export inference models with `export_torchscript.py`

### Getting Setup

#### Python Environment

I used a python `3.13.1` venv with the following installs:
```
pip install torch numpy joblib leidenalg matplotlib networkx psutil igraph scikit-learn tqdm tqdm_joblib uproot pyyaml scipy
```

#### Exporting Training Data

Use the `LArDLTwoDShowerGrowing` in training mode over some simulation, preferably with roll-up turned off. I would recommend a sample of around 300k events. Hadd all the outputted training data ROOT files together.

Take a look in `constants.py` to see if your detector is there. If not, you will need to add it to `DETECTORS` and provide scaling factors and pitches. To find the scaling factors you should use check inside your hadd'ed ROOT training data file (assuming you used the tree name "clusters")::
```
clusters_event_data->Scan("scalefactor_polar_r:scalefactor_cartesian_x:scalefactor_cartesian_z", "", "colsize=20 precision=7")
```
and for the pitches:
```
clusters_view_data->Scan("view:pitch", "", "colsize=20 precision=7")
```

Prepare a data directory for with an all/ directory:
```
my_dataset/
  all/
```
Use the script `data/read_cluster.py` to process the ROOT file into training data. Some notes on the flags:
- If you don't have any special requirements, set the hit feature vector preset to be 8, `--hit_feature_preset 8`
- To enable training with augmentations from iterative application of the prediction & clustering, you must have `--save_mc_cnts`
- For a mixed view training dataset, which is recommended, point `--out_dir_U`, `--out_dir_V`, and `--out_dir_W` all to the same directory
- If the data includes a large number of non-signal slices, use `--balance_events` to balance the signal (> 0.5 slice purity) and non-signal slices in the training dataset
- Specifying `--detector` will do confirm that the scaling factors and pitches in `constants.py` are the same as those in the TTree

The output will be a directory full of `.pt` files. Shuffle these into a directory structure like
```
my_dataset/
  train/
  val/
  test/
```

### Training Models

#### Running a Training

Trainings are configured through experiment yamls, see `experiments/example_experiment_config.yml`. They are submitted via the `train.py` script. The training will populate a directory in `checkpoints/` with loss values, validation examples, weight files etc. I like to have an rsync script to download all text and pdf files from the checkpoint dirs to check trainings.

For long trainings on systems with strict job time limits, the intended pattern is to chain together multiple experiment yamls. In practice this means enabling `save_latest_epoch_weights: True` in one experiment, then setting `continue_training_from_weights` in the next experiment to load those latest weights and continue from there. For example, you might run `my_experiment_continue0.yml`, then `my_experiment_continue1.yml` loading from `continue0`, then `my_experiment_continue2.yml` loading from `continue1`, and so on until the total number of epochs reaches the value you want. If you are training interactively or your cluster has unlimited walltime, you don't need to worry about this.

#### Evaluating a Training

`test_sim.py` can be used to evaluate the raw similarity predictions, `test_clustering.py` can be used to evaluate the clustering from the similarity prediction. `test_clustering.py` expects a separate holdout root file made by hadding the ouput of `LArDLTwoDShowerGrowing` ran in training mode. The results of the evaluation will be written to the checkpoint directory of the experiment.

#### Exporting to Torchscript

`export_to_torchscript.py` does what it says with the option to validate the export (this takes a long time). The flag `--use_chunked_similarity_forward` is recommended, this performs the similarity MLP inference is serial chunks, which is the correct thing to do for CPU inference. Memory usage can be very high if this is not set.

The following experiment yamls correspond to models that have been exported and are used in production code:

| Experiment YAML | Exported binaries |
| --- | --- |
| `Experiment_ShowerGrowing_DUNEFD_HD_v05_00_00.yml` | `PandoraNet_ShowerGrowing_DUNEFD_HD_Encoder_v05_00_00.pt`, `PandoraNet_ShowerGrowing_DUNEFD_HD_Attn_v05_00_00.pt`, `PandoraNet_ShowerGrowing_DUNEFD_HD_Sim_v05_00_00.pt` |
| `Experiment_ShowerGrowing_DUNEFD_VD_v05_00_00.yml` | `PandoraNet_ShowerGrowing_DUNEFD_VD_Encoder_v05_00_00.pt`, `PandoraNet_ShowerGrowing_DUNEFD_VD_Attn_v05_00_00.pt`, `PandoraNet_ShowerGrowing_DUNEFD_VD_Sim_v05_00_00.pt` |

### General Notes

- Iterative augmentations
  - This feature is controlled with `aug_params.iterative_augs: True/False` in the experiment config.
  - When enabled, augmentations are generated online during training and validation rather than being precomputed on disk. The model predicts a similarity matrix for the current event, a clustering step merges clusters using that prediction, and the merged clusters are then fed back through the model as higher augmentation tiers.
  - Iterative augmentations require training data produced with `data/read_cluster.py --save_mc_cnts`, since the MC ID counts are needed to rebuild the target similarity matrix after clusters have been merged.
  - The augmentation schedule is controlled by `aug_warmup_epoch`, `aug_freq_epoch`, and `aug_max_tier`. Training starts with no extra augmentation tiers, then the maximum allowed augmentation tier increases after the warmup period and continues to ramp up at the configured frequency.
  - Augmented examples are only produced when the clustering step actually merges something. If the predicted clustering leaves every input cluster separate, that event does not produce a higher-tier augmented version.
  - A consequence of this is that epoch length is variable when iterative augmentations are enabled. Later epochs are usually much longer because more higher-tier augmented passes are being added on top of the base training examples.
- Delayed additional hit features
  - The optional cardinality features (`hit_feat_add_cardinality`) and augmentation-tier feature (`hit_feat_add_aug_tier`) are intentionally held back for the first `hit_feat_add_at_epoch` epochs by rebuilding the dataloaders with those features disabled.
  - This was done because training was found to be more stable if the network first learns from the base hit features alone before these extra features are introduced.
  - If neither `hit_feat_add_cardinality` nor `hit_feat_add_aug_tier` is enabled, `hit_feat_add_at_epoch` is ignored.
- Learning-rate scheduling and training length
  - `OneCycleLR` was found to work well for this training.
  - A consequence of using this scheduler is that it is usually better to choose a predefined total number of epochs and let the training finish, rather than trying to use early stopping.
  - If the validation metrics show overtraining, the recommended approach is to rerun the experiment with a smaller total `epochs` value rather than stopping partway through the schedule.
- Training stability
  - The model can be sensitive to learning rate and can sometimes collapse if the rates are not chosen carefully.
  - In particular, the three subnetworks (`net_intra_cluster_encoder`, `net_inter_cluster_attn`, and `net_inter_cluster_sim`) may need different learning rates, so those values should be tuned with some care.
