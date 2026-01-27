import argparse, os, time, shutil, random

import logging; logger = logging.getLogger("the_logger")

import numpy as np

import torch; import torch.nn as nn
# NOTE Required to avoid "OSError: [Errno 24] Too many open file" when using dataloader multiprocessing
#      Pytorch issue #11201
torch.multiprocessing.set_sharing_strategy("file_system")

from config_parser import get_config
from helpers import setup_logging, get_gpu_usage, plot_pred_target, gen_augs
from dataset import ClusterDataset, ClusterMCCntsDataset, CollateClusters
from model import ClusterMergeNet
from clustering import connected_accessory_clustering

PRINT_N_PARAMS=True
PRINT_FORWARD_PASS=False
PRINT_CONF=True
N_VAL_PLOTS=5
N_TRAIN_PLOTS=5

# torch.manual_seed(1)

def main(args):
    conf = get_config(args.config_file)
    if PRINT_CONF:
        logger.info(f"Using config:\n{conf}")
    
    dataloader_train, dataloader_val = get_dataloaders(conf)
    steps_per_epoch = len(dataloader_train)
    epochs_per_step = 1 / len(dataloader_train)

    model = ClusterMergeNet(conf, steps_per_epoch=steps_per_epoch)
    model.train()
    if PRINT_N_PARAMS:
        model.print_num_params()
    if PRINT_FORWARD_PASS:
        model.print_forward_pass()

    epochs_per_val = epochs_per_step * conf.val_iter if conf.val_iter is not None else None
    next_val = model.training_start_epoch + epochs_per_val if conf.val_iter is not None else None

    max_aug_tier = 0
    collate_fn = CollateClusters(conf)
    n_hits_fn = lambda t_cluster: t_cluster.size(0)
    clustering_fn = (
        lambda pred_sim, clusters: connected_accessory_clustering(
            pred_sim, clusters, conf.aug_params["aug_sim_threshold"], n_hits_fn
        )
    )

    train_losses, best_val_loss = [], float("inf")
    max_gpu_mem = get_gpu_usage(model.device)
    for epoch in range(model.training_start_epoch, conf.epochs):
        time_s = time.time()
        curr_epoch = epoch
        if conf.aug_params["iterative_augs"]:
            new_max_aug_tier = max(
                (epoch - conf.aug_params["aug_warmup_epoch"]) // conf.aug_params["aug_freq_epoch"],
                0
            )
            if new_max_aug_tier != max_aug_tier:
                logger.info(f"Maximum aug tier increased: {max_aug_tier} -> {new_max_aug_tier}")
            max_aug_tier = new_max_aug_tier
            next_model_step = curr_epoch + epochs_per_step
        save_cntrs = [ 0 for _ in range(max_aug_tier + 1) ]

        model.step(new_epoch=curr_epoch)

        if (
            conf.hit_feat_add_at_epoch is not None and
            epoch >= conf.hit_feat_add_at_epoch and
            not dataloader_train.dataset.hit_feat_add_on
        ):
            dataloader_train, dataloader_val = get_dataloaders(conf, n_epoch=epoch)

        for data in dataloader_train:
            aug_tier = 0
            while True:
                model.set_input(data)
                model.optimize_parameters()
                update_train_loss(conf, train_losses, curr_epoch, model.get_loss())

                if conf.aug_params["aug_fixed_epoch_len"]:
                    curr_epoch += epochs_per_step * (len(data) / conf.batch_size)
                    if curr_epoch >= next_model_step:
                        model.step()
                        next_model_step += epochs_per_step

                if save_cntrs[aug_tier] < N_TRAIN_PLOTS:
                    ts = model.get_current_tensors()
                    out_dir = os.path.join(conf.train_dir_latest, f"{aug_tier}")
                    os.makedirs(out_dir, exist_ok=True)
                    save_cntrs[aug_tier] = plot_pred_target(
                        ts, out_dir, save_cntrs[aug_tier], N_TRAIN_PLOTS, conf
                    )

                if aug_tier >= max_aug_tier:
                    break
                if (
                    conf.aug_params["aug_proba"] is not None and
                    aug_tier == 0 and random.random() > conf.aug_params["aug_proba"]
                ):
                    break
                data = gen_augs(
                    data,
                    model.get_current_tensors()["ev_t_sim"],
                    clustering_fn,
                    collate_fn,
                    conf.aug_params["aug_max_n_jobs"],
                    aug_tier + 1,
                    conf,
                    dataset=dataloader_train.dataset
                )
                if data is None:
                    break
                aug_tier += 1

            if not conf.aug_params["aug_fixed_epoch_len"]:
                model.step()
                curr_epoch += epochs_per_step
            
            if next_val is not None and next_val <= curr_epoch:
                val_loss = val(conf, model, dataloader_val, max_aug_tier, clustering_fn, collate_fn)
                best_val_loss = update_val_loss(conf, val_loss, curr_epoch, best_val_loss, model)
                next_val += epochs_per_val

            if conf.aug_params["aug_fixed_epoch_len"] and curr_epoch > epoch + 1:
                break

        if get_gpu_usage(model.device) > max_gpu_mem:
            max_gpu_mem = get_gpu_usage(model.device)
            logger.info(f"New highest GPU usage: {max_gpu_mem:.2f} G")

        if next_val is None:
            val_loss = val(conf, model, dataloader_val, max_aug_tier, clustering_fn, collate_fn)
            best_val_loss = update_val_loss(conf, val_loss, curr_epoch, best_val_loss, model)

        logger.info(f"Epoch {epoch} complete - {int(time.time() - time_s) / 60:.1f} mins")

        if conf.save_latest_epoch_weights:
            os.makedirs(conf.weights_dir, exist_ok=True)
            model.save_networks(conf.latest_epoch_weights_filepath, epoch + 1, val_loss, conf.name)

def val(conf, model, dataloader, max_aug_tier, clustering_fn, collate_fn):
    model.eval()
    losses = []
    save_cntrs = [ 0 for _ in range(max_aug_tier + 1) ]
    epochs_per_step = 1 / len(dataloader)
    curr_epoch = 0
    for data in dataloader:
        aug_tier = 0
        while True:
            model.set_input(data)
            model.test(compute_loss=True)
            losses.append(model.get_loss())
            curr_epoch += epochs_per_step * (len(data) / conf.batch_size)

            if save_cntrs[aug_tier] < N_VAL_PLOTS:
                ts = model.get_current_tensors()
                out_dir = os.path.join(conf.val_dir_latest, f"{aug_tier}")
                os.makedirs(out_dir, exist_ok=True)
                save_cntrs[aug_tier] = plot_pred_target(
                    ts, out_dir, save_cntrs[aug_tier], N_VAL_PLOTS, conf
                )

            if aug_tier >= max_aug_tier:
                break
            if (
                conf.aug_params["aug_proba"] is not None and
                aug_tier == 0 and random.random() > conf.aug_params["aug_proba"]
            ):
                break
            data = gen_augs(
                data,
                model.get_current_tensors()["ev_t_sim"],
                clustering_fn,
                collate_fn,
                conf.aug_params["aug_max_n_jobs"],
                aug_tier + 1,
                conf,
                dataset=dataloader.dataset
            )
            if data is None:
                break
            aug_tier += 1

        if conf.aug_params["aug_fixed_epoch_len"] and curr_epoch > 1:
            break

    model.train()

    return np.mean(losses)

def update_val_loss(conf, loss, curr_epoch, best_val_loss, model):
    with open(conf.loss_file, "a+") as f:
        f.write(f"VALID {curr_epoch:.6f} {loss:.6f}\n")
    logger.info(f"Epoch {curr_epoch:.2f} - Validation loss: {loss:.6f}")

    if loss < best_val_loss:
        logger.info(f"New best val loss: {best_val_loss:.6f} -> {loss:.6f}")
        best_val_loss = loss
        if os.path.exists(conf.val_dir_best):
            shutil.rmtree(conf.val_dir_best)
        shutil.copytree(conf.val_dir_latest, conf.val_dir_best)
        if conf.save_best_weights:
            os.makedirs(conf.weights_dir, exist_ok=True)
            model.save_networks(conf.best_weights_filepath, curr_epoch, loss, conf.name)

    return best_val_loss

def update_train_loss(conf, losses, curr_epoch, new_loss):
    losses.append(new_loss)
    if len(losses) >= conf.train_loss_iter:
        loss = np.mean(losses)
        losses.clear()
        with open(conf.loss_file, "a+") as f:
            f.write(f"TRAIN {curr_epoch:.6f} {loss:.6f}\n")
        logger.info(f"Epoch {curr_epoch:.2f} - Training loss: {loss:.6f}")

def get_dataloaders(conf, n_epoch=0):
    collate_fn = CollateClusters(conf)

    dataset_cls = ClusterMCCntsDataset if conf.aug_params["iterative_augs"] else ClusterDataset

    dataset_train = dataset_cls(os.path.join(conf.data_path, "train"), conf)
    logger.info(f"{len(dataset_train)} training samples")
    dataset_val = dataset_cls(os.path.join(conf.data_path, "val"), conf)
    if conf.hit_feat_add_at_epoch is not None and conf.hit_feat_add_at_epoch > n_epoch:
        dataset_train.hit_feat_add_on = False
        dataset_val.hit_feat_add_on = False
    logger.info(f"{len(dataset_val)} validation samples")

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=conf.batch_size,
        collate_fn=collate_fn,
        num_workers=min(conf.max_num_workers, conf.batch_size),
        shuffle=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=conf.batch_size,
        collate_fn=collate_fn,
        num_workers=min(conf.max_num_workers, conf.batch_size),
        shuffle=True
    )

    return dataloader_train, dataloader_val

def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    setup_logging()
    main(parse_cli())
