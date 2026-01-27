""" Reading and processing experiment config yaml into a namedtuple """

import os, shutil
from collections import namedtuple

import logging; logger = logging.getLogger("the_logger")

import yaml

""" Start - config parser for cluster similarity prediciton """

COMMON_DEFAULTS={
    "device" : "cuda:0",
    "max_num_workers" : 4,
    "optimizer": "Adam",
    "lr_scheduler_params": { },
    "lr_scheduler" : None,
    "gradient_clipping" : True,
    "train_loss_iter" : 100,
    "val_iter" : None, # None is each epoch
    "loss_params" : { } ,
    "amp_training" : True,
    "plot_params" : { },
    "save_best_weights" : False,
    "save_latest_epoch_weights" : False,
    "continue_training_from_weights" : None
}

COMMON_MANDATORY_FIELDS={
    "checkpoints_dir",
    "data_path",
    "batch_size",
    "epochs",
    "name"
}

CLUSTER_SIM_DEFAULTS=COMMON_DEFAULTS | {
    "lr_scheduler_params": {
        "scheduler" : None,
        "max_lr_factor" : None
    },
    "hit_feat_vec_dim" : 5,
    "hit_feat_scaling_factors" : {},
    "hit_feat_log_transforms" : [],
    "hit_feat_add_cardinality": False,
    "hit_feat_add_aug_tier": False,
    "hit_feat_add_at_epoch": 2,
    "net_intra_cluster_encoder_params" : {
        "embd_dim" : 120,
        "num_heads" : 5,
        "hidden_dim" : 240,
        "num_inds" : 32,
        "depth" : 2,
        "ln" : False,
        "lr" : 1e-5
    },
    "net_inter_cluster_attn_params" : {
        "embd_dim" : 120,
        "num_heads" : 5,
        "hidden_dim" : 240,
        "num_inds" : 32,
        "depth" : 2,
        "ln" : False,
        "lr" : 1e-4
    },
    "net_inter_cluster_sim_params" : {
        "hidden_dim" : 128,
        "lr" : 1e-4,
        "use_loop_implementation" : False,
        "use_chunked_implementation" : False,
        "chunk_size" : 1024
    },
    "bucket_boundaries" : [4, 8, 32, 128],
    "loss_params" : {
        "weights" : None,
        "triplet_lambda" : 0.,
        "triplet_n_samples" : 50000,
        "contrastive_lambda" : 0.
    },
    "plot_params" : {
        "has_summary_token" : False,
        "polar_coords" : False,
        "view" : 6, # -ve to infer from feature vec one-hot view encoding, number is index of last element of encoding
        "x_width_idx" : 2
    },
    "save_best_weights" : False,
    "save_latest_epoch_weights" : False,
    "aug_params" : {
        "iterative_augs" : False,
        "aug_sim_threshold" : 0.5,
        "aug_freq_epoch" : 1,
        "aug_warmup_epoch" : 0,
        "aug_max_n_jobs" : 0,
        "aug_fixed_epoch_len": False,
        "aug_proba": None
    }
}

CLUSTER_SIM_MANDATORY_FIELDS=COMMON_MANDATORY_FIELDS | set()

def get_config(conf_file, overwrite_dict={}, test=False):
    logger.info(f"Reading conf from {conf_file}")

    conf_dict = _parse_yaml(
        conf_file, CLUSTER_SIM_MANDATORY_FIELDS, CLUSTER_SIM_DEFAULTS, overwrite_dict
    )

    if test:
        conf_dict["lr_scheduler_params"]["scheduler"] = None
    else:
        _prep_checkpoints_dir(conf_dict, conf_file)

    if "hit_feat_add_cardinality_epoch" in conf_dict:
        conf_dict["hit_feat_add_at_epoch"] = conf_dict["hit_feat_add_cardinality_epoch"]
        del conf_dict["hit_feat_add_cardinality_epoch"]
    if conf_dict["hit_feat_add_aug_tier"] and not conf_dict["aug_params"]["iterative_augs"]:
        raise ValueError("'hit_feat_add_aug_tier' only makes sense with iterative augs enabled")
    if not conf_dict["hit_feat_add_cardinality"] and not conf_dict["hit_feat_add_aug_tier"]:
        conf_dict["hit_feat_add_at_epoch"] = None

    conf_namedtuple = namedtuple("conf", conf_dict)
    conf = conf_namedtuple(**conf_dict)

    return conf

""" End - config parser for cluster similarity prediction """

""" Start - helpers """

def _parse_yaml(conf_file, mandatory_fields, defaults, overwrite_dict):
    with open(conf_file) as f:
        conf_dict = yaml.load(f, Loader=yaml.FullLoader)

    missing_fields = mandatory_fields - set(conf_dict.keys())
    if missing_fields:
        raise ValueError(
            f"Missing mandatory fields {missing_fields} in conf file at {conf_file}"
        )

    for option, val in conf_dict.items(): # handle once-nested dictionary
        if isinstance(val, dict):
            for key in set(defaults[option].keys()) - set(conf_dict[option].keys()):
                conf_dict[option][key] = defaults[option][key]
    for option in set(defaults.keys()) - set(conf_dict.keys()):
        conf_dict[option] = defaults[option]

    for field, val in overwrite_dict.items():
        if isinstance(field, tuple):
            assert len(field) == 2
            conf_dict[field[0]][field[1]] = val
        else:
            conf_dict[field] = val

    return conf_dict

def _prep_checkpoints_dir(conf_dict, conf_file):
    conf_dict["checkpoint_dir"] = os.path.join(conf_dict["checkpoints_dir"], conf_dict["name"])
    if not os.path.exists(conf_dict["checkpoint_dir"]):
        os.makedirs(conf_dict["checkpoint_dir"])
    else:
        logger.warning(f"{conf_dict['checkpoint_dir']} already exists, data may be overwritten")
    shutil.copyfile(
        conf_file, os.path.join(conf_dict["checkpoint_dir"], os.path.basename(conf_file))
    )

    n = 0
    while (
        os.path.exists(os.path.join(conf_dict["checkpoint_dir"], f"losses_{n}.txt")) or
        os.path.exists(os.path.join(conf_dict["checkpoint_dir"], f"val_{n}/")) or 
        os.path.exists(os.path.join(conf_dict["checkpoint_dir"], f"train_{n}/")) or
        os.path.exists(os.path.join(conf_dict["checkpoint_dir"], f"weights_{n}/"))
    ):
        n += 1
    conf_dict["loss_file"] = os.path.join(conf_dict["checkpoint_dir"], f"losses_{n}.txt")
    logger.info(f"loss file will be {os.path.basename(conf_dict["loss_file"])}")

    conf_dict["val_dir"] = os.path.join(conf_dict["checkpoint_dir"], f"val_{n}/")
    conf_dict["val_dir_latest"] = os.path.join(conf_dict["val_dir"], "latest/")
    conf_dict["val_dir_best"] = os.path.join(conf_dict["val_dir"], "best/")
    logger.info(f"val predictions dir will be {conf_dict["val_dir"]}")

    conf_dict["train_dir"] = os.path.join(conf_dict["checkpoint_dir"], f"train_{n}/")
    conf_dict["train_dir_latest"] = os.path.join(conf_dict["train_dir"], "latest/")
    logger.info(f"train predictions dir will be {conf_dict["train_dir"]}")

    if conf_dict["save_best_weights"] or conf_dict["save_latest_epoch_weights"]:
        conf_dict["weights_dir"] = os.path.join(conf_dict["checkpoint_dir"], f"weights_{n}/")
        logger.info(f"checkpointed weights dir will be {conf_dict["weights_dir"]}")
    if conf_dict["save_best_weights"]:
        conf_dict["best_weights_filepath"] = os.path.join(
            conf_dict["weights_dir"], "best_weights.pt"
        )
    if conf_dict["save_latest_epoch_weights"]:
        conf_dict["latest_epoch_weights_filepath"] = os.path.join(
            conf_dict["weights_dir"], "latest_epoch_weights.pt"
        )

""" End - helpers """
