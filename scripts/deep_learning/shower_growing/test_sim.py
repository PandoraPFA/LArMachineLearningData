""" Evaluate the predicted/target similarity matrix """

import argparse, os, functools
from collections import defaultdict
import warnings; warnings.filterwarnings("ignore") # sklearn being very pushy

import logging; logger = logging.getLogger("the_logger")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, accuracy_score
)

import torch; import torch.nn as nn
# NOTE Required to avoid "OSError: [Errno 24] Too many open file" when using dataloader multiprocessing
#      Pytorch issue #11201
torch.multiprocessing.set_sharing_strategy("file_system")

from config_parser import get_config
from helpers import (
    setup_logging,
    get_gpu_usage,
    plot_pred_target,
    plot_pred_target_adj,
    unscale_cluster_tensor_inplace,
    get_view,
    get_view_str,
    gen_augs
)
from clustering import connected_accessory_clustering
from dataset import ClusterDataset, ClusterMCCntsDataset, CollateClusters
from model import ClusterMergeNet

torch.manual_seed(1)

PRINT_N_PARAMS=True
PRINT_FORWARD_PASS=False
N_TEST_PLOTS=20

def main(args):
    overwrite_dict = {}
    if args.iterative_augs:
        overwrite_dict[("aug_params", "iterative_augs")] = True
    if args.overwrite_data_path is not None:
        overwrite_dict["data_path"] = args.overwrite_data_path
    conf = get_config(args.config_file, test=True, overwrite_dict=overwrite_dict)

    dataloader_test = get_dataloader(conf, args.max_events)

    model = prep_model(conf, args.weights_file)

    collate_fn = CollateClusters(conf)
    n_hits_fn = lambda t_cluster: t_cluster.size(0)
    clustering_fn = (
        lambda pred_sim, clusters: connected_accessory_clustering(
            pred_sim, clusters, args.adjacency_threshold, n_hits_fn
        )
    )

    test_dir = os.path.join(
        os.path.dirname(args.weights_file), "test_iter_augs" if args.iterative_augs else "test"
    )
    test_dirs = { 0 : test_dir, 1 : os.path.join(test_dir, "1"), 2 : os.path.join(test_dir, "2+") }
    prep_test_dir(test_dirs, args.iterative_augs)

    y_sim_preds = { 0 : defaultdict(list), 1 : defaultdict(list), 2 : defaultdict(list) }
    y_sim_targets = { 0 : defaultdict(list), 1 : defaultdict(list), 2 : defaultdict(list) }
    n_events = { 0 : defaultdict(int), 1 : defaultdict(int), 2 : defaultdict(int) }
    losses = []
    test_plot_cntr = { 0 : 0, 1 : 0, 2 : 0 }
    for data in tqdm(dataloader_test, desc="Processing test data..."):
        aug_tier = 0
        while True:
            i = min(aug_tier, 2)

            model.set_input(data)
            model.test(compute_loss=True)
            losses.append(model.get_loss())

            view = conf.plot_params["view"]
            if args.UVW:
                view = get_view(view, data["chunked_input"][0][0][0])
            view = get_view_str(view)
            n_events[i][view] += 1

            # Collate predictions for each individual cluster association
            ts = model.get_current_tensors()
            t_sim_pred, t_sim_target = ts["ev_t_sim"][0][0], ts["ev_t_sim_target"][0][0]
            N = t_sim_pred.size(0)
            idx_i, idx_j = torch.triu_indices(N, N, offset=1)
            y_sim_preds[i][view].append(
                t_sim_pred[idx_i, idx_j].float().detach().cpu().numpy().ravel()
            )
            y_sim_targets[i][view].append(
                t_sim_target[idx_i, idx_j].float().detach().cpu().numpy().ravel()
            )

            if test_plot_cntr[i] < N_TEST_PLOTS:
                test_plot_cntr[i] += 1

                for t_chunk_cluster in ts["chunked_t_clusters"]:
                    for t_cluster in t_chunk_cluster:
                        unscale_cluster_tensor_inplace(
                            t_cluster, conf.hit_feat_scaling_factors, conf.hit_feat_log_transforms
                        )

                with PdfPages(
                    os.path.join(test_dirs[i], f"{test_plot_cntr[i]}_test_sim.pdf")
                ) as pdf:
                    plot_pred_target(ts, None, None, None, conf, unscale=False)
                    pdf.savefig()
                    plt.close()

                    plot_pred_target_adj(
                        t_sim_pred, (t_sim_pred >= args.adjacency_threshold),
                        t_sim_target, (t_sim_target >= args.adjacency_threshold),
                        args.adjacency_threshold, None
                    )
                    pdf.savefig()
                    plt.close()

            if not args.iterative_augs:
                break

            data = gen_augs(
                data,
                model.get_current_tensors()["ev_t_sim"],
                clustering_fn, collate_fn,
                0, aug_tier + 1, conf,
                dataset=dataloader_test.dataset
            )
            if data is None:
                break
            aug_tier += 1

    logger.info(f"Max GPU usage during test: {get_gpu_usage(model.device):.2f} G")

    loss = np.mean(losses)
    logger.info(f"Mean loss: {loss:.6f}")

    for i in (0, 1, 2):
        y_sim_pred = { view : np.concatenate(y) for view, y in y_sim_preds[i].items() }
        y_sim_target = { view : np.concatenate(y) for view, y in y_sim_targets[i].items() }
        y_adj_pred = {
            view : (y >= args.adjacency_threshold).astype(int) for view, y in y_sim_pred.items()
        }
        y_adj_target = {
            view : (y >= args.adjacency_threshold).astype(int) for view, y in y_sim_target.items()
        }

        metrics_file = os.path.join(test_dirs[i], "metrics_test_sim.txt")
        if os.path.exists(metrics_file):
            os.remove(metrics_file)
        for view in y_sim_pred:
            report_metrics(
                y_sim_pred[view], y_sim_target[view],
                y_adj_pred[view], y_adj_target[view],
                n_events[i][view],
                view, metrics_file
            )

def report_metrics(y_sim_pred, y_sim_target, y_adj_pred, y_adj_target, n_events, view, metrics_file):
    logger.info(f"-- Metrics for view {view} events in test set, total {n_events}")

    logger.info(f"Metrics per element of similarity matrix:")
    mae = mean_absolute_error(y_sim_target, y_sim_pred)
    logger.info(f"MAE: {mae:.6f}")
    mse = mean_squared_error(y_sim_target, y_sim_pred)
    logger.info(f"MSE: {mse:.6f}")

    logger.info(f"Adjacency matrices made using similarity theshold of {args.adjacency_threshold}")
    logger.info("Metrics per element of adjacency matrix:")
    acc = accuracy_score(y_adj_target, y_adj_pred)
    logger.info(f"Acc.: {acc:.4f}")
    balanced_acc = balanced_accuracy_score(y_adj_target, y_adj_pred)
    logger.info(f"Balanced acc.: {balanced_acc:.4f}")
    precision = precision_score(y_adj_target, y_adj_pred)
    logger.info(f"Precision: {precision:.4f}")
    recall = recall_score(y_adj_target, y_adj_pred)
    logger.info(f"Recall: {recall:.4f}")
    f1 = f1_score(y_adj_target, y_adj_pred)
    logger.info(f"F1 score: {f1:.4f}")

    with open(metrics_file, "a") as f:
        f.write(f"{view} n_events {n_events}\n")
        f.write(f"{view} mae {mae:.6f}\n")
        f.write(f"{view} mse {mse:.6f}\n")
        f.write(f"{view} acc {acc:.4f}\n")
        f.write(f"{view} bal_acc {balanced_acc:.4f}\n")
        f.write(f"{view} prec {precision:.4f}\n")
        f.write(f"{view} rec {recall:.4f}\n")
        f.write(f"{view} f1 {f1:.4f}\n")

def prep_model(conf, weights_file):
    model = ClusterMergeNet(conf)
    model.eval()
    model.load_networks(weights_file)
    if PRINT_N_PARAMS:
        model.print_num_params()
    if PRINT_FORWARD_PASS:
        model.print_forward_pass()

    return model

def prep_test_dir(test_dirs, iterative_augs):
    if not os.path.exists(test_dirs[0]):
        os.makedirs(test_dirs[0])
    else:
        logger.warning(f"{test_dirs[0]} already exists, data may be overwritten")
    if iterative_augs:
        for test_dir in (test_dirs[1], test_dirs[2]):
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

def get_dataloader(conf, max_events):
    dataset_cls = ClusterMCCntsDataset if conf.aug_params["iterative_augs"] else ClusterDataset
    collate_fn = CollateClusters(conf)
    dataset_test = dataset_cls(os.path.join(conf.data_path, "test"), conf, max_events=max_events)
    logger.info(f"{len(dataset_test)} test samples")
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, collate_fn=collate_fn, num_workers=0, shuffle=True
    )
    return dataloader_test

def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file")
    parser.add_argument("weights_file")

    parser.add_argument("--adjacency_threshold", type=float, default=0.5)
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--batch_mode", action="store_true")
    parser.add_argument("--UVW", action="store_true")
    parser.add_argument("--iterative_augs", action="store_true")
    parser.add_argument("--overwrite_data_path", type=str, default=None)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    setup_logging()
    args = parse_cli()
    if args.batch_mode:
        tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=True)
    main(args)

