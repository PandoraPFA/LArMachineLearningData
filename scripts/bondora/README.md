## BOndora - Bayesian Optimisation of Pandora Algorithms

BOndora (PandoBO, Bo'Pandora, Bandora, ...) is used to perform Bayesian Optimisation to maximise the performance of Pandora workflows over a training dataset of `pndr` files. The optimisation is with respect to a collection of algorithm configuration parameters (leaf nodes of the Pandora workflow xml). It was developed to optimise the initial 2D clustering stage of Pandora but is applicable more broadly.

BOndora is basically a wrapper of [Optuna](https://github.com/optuna/optuna) to allow Optuna's Bayesian Optimisation implementation (`GPSampler`) to talk to Pandora.

### Getting Setup

#### Python environment

I was using a python `3.9.15` venv with the following:
```
pip install optuna numpy matplotlib scipy sympy pyyaml psutil torch
```
I expect any python 3 with the above libraries installed will work. I had optuna `4.2.1`.

You should make a bash script to setup this venv, here's mine:
```
$ cat setup_venv.sh 
source /cvmfs/larsoft.opensciencegrid.org/spack-packages/setup-env.sh
spack load python@3.9.15
source /storage/epp2/phsajw/bondora/.venv_3.9.15_optuna/bin/activate
```

#### Prepare Pandora Build

You will need a build of pandora accessible. LArContent needs to be at tag >= v04_16_01 to have the validation algorithm Used in my optimisation studies.

#### Prepare Directories

You will need a `scratch/` and a `study_results/` directory somewhere.

#### Prepare Datasets

You will need a train and a test dataset. This should be a directory filled with `.pndr` files. 

### Defining a Study

A study is one optimisation pass that consists of many trials at different configuration parameter points.

You will need to put a Pandora workflow xml in `settings/` that should have all the algorithms you want to include in your optimisation. Any algorithm configuration parameters you want to optimise for should be included explicitly in the xml. The xml should end with the `LArEventClusterValidation` algorithm. You are welcome to use any validation algorithm though, the only requirement is that it writes out a TTree (that you will have to parse). There are some example xmls in `settings/2d_clustering`:
```
Pandora_Neutrino_DUNEFD_W.xml            - Optimise the 2D clustering W view parameters w.r.t.
                                           performance immediately after the 2D clustering
Pandora_Neutrino_DUNEFD_UV.xml           - Optimise the 2D clustering U & V plane parameters
                                           jointly w.r.t. performance immediately after the
                                           2D clustering
Pandora_Neutrino_DUNEFD_fullreco_UVW.xml - Optimise the 2D clustering U & V & W plane parameters
                                           jointly w.r.t. performance at the end of the
                                           standard Pandora neutrino workflow
```

Decide what metrics you want to extract from the validation algorithm to form an objective value with back in python. A script like `scripts/aggregate_validations_*` is used to hadd all the validation TTrees and then extract average metric values (with cuts if desired) into a plain text file. This plain text file is then parsed with one of the derived classes in `results_parsers.py`. So make sure you are happy with one of the existing aggregate scripts and results parsers or write your own aggregate script/results parser class. If you decide to use a validation alg other than `EventClusterValidation`, you will definitely need your own aggregate script and results parser.

You will then need to configure your study using a yaml file which gets parsed by `config_parser.py`. See `study_configs/2dclustering_W_example.yml` for a commented example. This yaml defines a study that optimises the 2D clustering W view parameters w.r.t. a combinations of ARI, track purity, and track completeness. It runs on a 48 core server for 400 BO trials over 600 pndr files. It takes a bit under 2 days. A lot of it is just pointing to files and directories the study will use, you will need to change all these paths.. The parameters that require some thought are:
```
n_processes - Set this to as many cpu cores you have in the job
n_files_total - The number of files to run for a single trial, the more the better but slower
expected_single_file_process_time - A rough estimate of how long Pandora should be taking for one pndr,
                                    if this exceeded greatly, the trial is terminated and the
                                    final objective is penalised
search_space - This the most important! Defines the dimensions and bounds of the search space
               the Gaussian Process tries to fit
initial_param_point - This sets the first configuration parameter point of the study,
                      I recommend setting this to the default configuration values
```

### Running a Study

You can now run your study. `bo_pndr.py` is the entry point:
```
python bo_pndr.py study_configs/my_cool_study_that_will_save_the_world.yml
```

You can just run this interactively if you have access to a machine with many cores. Or you can submit it to a grid, I was using slurm `slurm/run_bo.sh`.

The output of the study is stored in Optuna's DB file. You can inspect the results with the `bo_eval.py` script, for example:
```
python bo_eval.py --has_default_trial --get_best_trial --plot_history study_configs/my_cool_study_that_will_save_the_world.yml
```

There are many options for other visualisation and exporting the optimised parameters to a new xml.

### Bayesian Optimisation Advice

- BO works pretty well up to ~20 parameters in the search space. If you have more parameters than this, you can select the ~20 most important parameters by:
    1. Run a study with the "random" sampler using all your parameters. This study should use less pndr files so it can be
       ran for >1000 trials.
    2. Get the most important parameters of this random optimisation with the
       `--get_param_importance` and `--plot_param_importance` flags for `bo_eval.py`
    3. Read off the top ~20 parameters
    4. Do you "gp" sampler study on this subset of the most important parameters

- It is important that the BO is performed with a training dataset distinct from the data used in any evaluation of the new configuration parameters in Pandora. It is very possible to overtrain.

### Potential Problems

- The configuration parameter will be replaced everywhere it appears in the xml with the same algorithm type and xml tag signature in the search space. This might be undesirable, `suggest_and_set_params` will need to be updated in `bo_pndr.py` to distinguish between identical blocks appearing at different stages somehow.
