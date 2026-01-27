""" Evaluate the cluster merges derived from the predicted/target similarity matrix """

import argparse, os, functools
from collections import defaultdict, Counter
import warnings; warnings.filterwarnings("ignore") # sklearn being very pushy

import logging; logger = logging.getLogger("the_logger")

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import uproot
import numpy as np
from tqdm import tqdm

import torch
# NOTE Required to avoid "OSError: [Errno 24] Too many open file" when using dataloader multiprocessing
#      Pytorch issue #11201
torch.multiprocessing.set_sharing_strategy("file_system")

from sklearn.metrics.cluster import adjusted_rand_score, contingency_matrix

from config_parser import get_config
from model import ClusterMergeNet
from dataset import CollateClusters
from data.read_cluster import read_events, make_cluster_data, get_similarity_matrix
from helpers import (
    setup_logging,
    get_gpu_usage,
    plot_clusterings,
    plot_pred_target_adj,
    scale_cluster_tensor_inplace,
    add_cardinality_feature,
    add_aug_tier_feature,
    plot_clusters,
    gen_single_aug
)
from clustering import (
    affinity_clustering,
    leiden_clustering,
    agglomerative_clustering,
    connected_accessory_clustering,
    connected_accessory_clustering_2stage,
    connected_clustering
)

torch.manual_seed(1)

VIEW_CONVERSION = { "U" : 4, "V" : 5, "W" : 6 }
PRINT_N_PARAMS=True
PRINT_FORWARD_PASS=False
N_TEST_PLOTS=50
N_HITS_THRES=10

def main(args):
    view = VIEW_CONVERSION[args.view]

    conf = get_config(args.config_file, test=True)

    model = prep_model(conf, args.weights_file)

    collate_fn = CollateClusters(conf)
    if args.clustering_mode != 7:
        collate_fn.include_cluster_mc_id_cnt = False

    suffix = "_" + args.test_dir_suffix if args.test_dir_suffix else ""
    test_dir = os.path.join(os.path.dirname(args.weights_file), "test" + suffix)
    prep_test_dir(test_dir)

    n_hits = []
    hit_perfect_true_aris, hit_perfect_true_purities, hit_perfect_true_completenesses = [], [], []
    hit_target_true_aris, hit_target_true_purities, hit_target_true_completenesses = [], [], []
    hit_pred_true_aris, hit_pred_true_purities, hit_pred_true_completenesses = [], [], []
    hit_pred_target_aris = []

    n_hits_baseline = []
    hit_baseline_pred_true_aris = []
    hit_baseline_pred_true_purities, hit_baseline_pred_true_completenesses = [], []
    if args.baseline_root_file is not None:
        events_baseline = read_events(uproot.open(args.baseline_root_file)[args.root_treename])#[13:14]

    test_plot_cntr = 0
    events = read_events(uproot.open(args.root_file)[args.root_treename], n_events=args.max_events)#[13:14]
    for i_event, event in tqdm(
        enumerate(events), desc="Processing test events...", total=len(events)
    ):
        clusters = event.view_clusters[view]
        for cluster in clusters:
            cluster.calc_main_mc()
        if args.clustering_mode == 7: # Iterative clustering logic too tricky to implement nicely
            hit_labels_target = iterative_clustering(
                model, clusters, event, view, collate_fn, conf, args, use_target=True
            )
            hit_labels_pred = iterative_clustering(
                model, clusters, event, view, collate_fn, conf, args
            )
        else:
            t_sim_pred, t_sim_target = get_pred_target_sim(
                model, clusters, event, view, args.hit_feature_preset, collate_fn, conf
            )
            hit_labels_target = cluster_from_similarity(t_sim_target, clusters, args)
            hit_labels_pred = cluster_from_similarity(t_sim_pred, clusters, args)

        hit_labels_true = get_true_clusterings(clusters)
        hit_labels_perfect = get_perfect_clusterings(clusters)

        metrics_mask = (hit_labels_true != -1) # Hits with missing MC info
        n_hits.append(int(metrics_mask.sum()))

        ret = calc_metrics(hit_labels_true[metrics_mask], hit_labels_target[metrics_mask])
        hit_target_true_aris.append(ret[0])
        hit_target_true_purities.append(ret[1])
        hit_target_true_completenesses.append(ret[2])

        ret = calc_metrics(hit_labels_true[metrics_mask], hit_labels_perfect[metrics_mask])
        hit_perfect_true_aris.append(ret[0])
        hit_perfect_true_purities.append(ret[1])
        hit_perfect_true_completenesses.append(ret[2])

        ret = calc_metrics(hit_labels_true[metrics_mask], hit_labels_pred[metrics_mask])
        hit_pred_true_aris.append(ret[0])
        hit_pred_true_purities.append(ret[1])
        hit_pred_true_completenesses.append(ret[2])

        hit_pred_target_aris.append(
            calc_metrics(
                hit_labels_target[metrics_mask], hit_labels_pred[metrics_mask], only_ari=True
            )
        )

        if args.baseline_root_file is not None:
            clusters_baseline = events_baseline[i_event].view_clusters[view]
            for cluster in clusters_baseline:
                cluster.calc_main_mc()
            hit_labels_baseline_pred = np.array(
                [ label for label, cluster in enumerate(clusters_baseline) for _ in cluster.hits ]
            )
            hit_labels_baseline_true = get_true_clusterings(clusters_baseline)
            metrics_mask = (hit_labels_baseline_true != -1)
            if metrics_mask.sum() == 0: # No 2D hits made it to 3D, results in no 2D clusters
                hit_labels_baseline_pred = None
                continue
            n_hits_baseline.append(int(metrics_mask.sum()))
            ret = calc_metrics(
                hit_labels_baseline_true[metrics_mask], hit_labels_baseline_pred[metrics_mask]
            )
            hit_baseline_pred_true_aris.append(ret[0])
            hit_baseline_pred_true_purities.append(ret[1])
            hit_baseline_pred_true_completenesses.append(ret[2])

        if test_plot_cntr < N_TEST_PLOTS:
            test_plot_cntr += 1

            with PdfPages(os.path.join(test_dir, f"{test_plot_cntr}_test_cluster.pdf")) as pdf:
                plot_clusterings(
                    [ hit for cluster in clusters for hit in cluster.hits ],
                    [ label for label, cluster in enumerate(clusters) for _ in cluster.hits ],
                    hit_labels_pred,
                    hit_labels_true,
                    None,
                    conf,
                    baseline_labels=(
                        hit_labels_baseline_pred if args.baseline_root_file is not None else None
                    ),
                    # target_labels=hit_labels_target
                    target_labels=None 
                )
                pdf.savefig()
                plt.close()

                if args.clustering_mode != 7:
                    plot_pred_target_adj(
                        t_sim_pred, (t_sim_pred >= args.sim_threshold),
                        t_sim_target, (t_sim_target >= args.sim_threshold),
                        args.sim_threshold,
                        None
                    )
                    pdf.savefig()
                    plt.close()

    logger.info(f"Max GPU usage during test: {get_gpu_usage(model.device):.2f} G")

    n_hits_mask = (np.array(n_hits) > N_HITS_THRES)
    hit_perfect_true_aris = np.array(hit_perfect_true_aris)[n_hits_mask]
    hit_perfect_true_purities = np.array(hit_perfect_true_purities)[n_hits_mask]
    hit_perfect_true_completenesses = np.array(hit_perfect_true_completenesses)[n_hits_mask]
    hit_target_true_aris = np.array(hit_target_true_aris)[n_hits_mask]
    hit_target_true_purities = np.array(hit_target_true_purities)[n_hits_mask]
    hit_target_true_completenesses = np.array(hit_target_true_completenesses)[n_hits_mask]
    hit_pred_true_aris = np.array(hit_pred_true_aris)[n_hits_mask]
    hit_pred_true_purities = np.array(hit_pred_true_purities)[n_hits_mask]
    hit_pred_true_completenesses = np.array(hit_pred_true_completenesses)[n_hits_mask]
    hit_pred_target_aris = np.array(hit_pred_target_aris)[n_hits_mask]

    logger.info(f"{n_hits_mask.sum()} total events for perfect/target/pred")
    logger.info(
        "Perfect Clusters:\n"
        f" - ARI          {np.mean(hit_perfect_true_aris):.4f}\n"
        f" - Purity       {np.mean(hit_perfect_true_purities):.4f}\n"
        f" - Completeness {np.mean(hit_perfect_true_completenesses):.4f}"
    )
    logger.info(
        "Target Clusters:\n"
        f" - ARI          {np.mean(hit_target_true_aris):.4f}\n"
        f" - Purity       {np.mean(hit_target_true_purities):.4f}\n"
        f" - Completeness {np.mean(hit_target_true_completenesses):.4f}"
    )
    logger.info(
        "Pred Clusters:\n"
        f" - ARI          {np.mean(hit_pred_true_aris):.4f}\n"
        f" - Purity       {np.mean(hit_pred_true_purities):.4f}\n"
        f" - Completeness {np.mean(hit_pred_true_completenesses):.4f}"
    )
    logger.info(f"Pred-Target Hit ARI: {np.mean(hit_pred_target_aris):.4f}")

    if args.baseline_root_file is not None:
        n_hits_baseline_mask = (np.array(n_hits_baseline) > N_HITS_THRES)
        hit_baseline_pred_true_aris = (
            np.array(hit_baseline_pred_true_aris)[n_hits_baseline_mask]
        )
        hit_baseline_pred_true_purities = (
            np.array(hit_baseline_pred_true_purities)[n_hits_baseline_mask]
        )
        hit_baseline_pred_true_completenesses = (
            np.array(hit_baseline_pred_true_completenesses)[n_hits_baseline_mask]
        )

        logger.info(f"{n_hits_baseline_mask.sum()} total events for baseline")
        logger.info(
            "Baseline Reco Clusters:\n"
            f" - ARI          {np.mean(hit_baseline_pred_true_aris):.4f}\n"
            f" - Purity       {np.mean(hit_baseline_pred_true_purities):.4f}\n"
            f" - Completeness {np.mean(hit_baseline_pred_true_completenesses):.4f}"
        )

    metrics_file = os.path.join(test_dir, "metrics_test_clustering.txt")
    with open(metrics_file, "w") as f:
        f.write(f"n events {n_hits_mask.sum()}\n")
        f.write(f"perfect ari {np.mean(hit_perfect_true_aris):.4f}\n")
        f.write(f"perfect purity {np.mean(hit_perfect_true_purities):.4f}\n")
        f.write(f"perfect completeness {np.mean(hit_perfect_true_completenesses):.4f}\n")
        f.write(f"target ari {np.mean(hit_target_true_aris):.4f}\n")
        f.write(f"target purity {np.mean(hit_target_true_purities):.4f}\n")
        f.write(f"target completeness {np.mean(hit_target_true_completenesses):.4f}\n")
        f.write(f"pred ari {np.mean(hit_pred_true_aris):.4f}\n")
        f.write(f"pred purity {np.mean(hit_pred_true_purities):.4f}\n")
        f.write(f"pred completeness {np.mean(hit_pred_true_completenesses):.4f}\n")
        f.write(f"pred-target ari {np.mean(hit_pred_target_aris):.4f}\n")

        if args.baseline_root_file is not None:
            f.write(f"baseline reco ari {np.mean(hit_baseline_pred_true_aris):.4f}\n")
            f.write(f"baseline reco purity {np.mean(hit_baseline_pred_true_purities):.4f}\n")
            f.write(f"baseline reco completeness {np.mean(hit_baseline_pred_true_completenesses):.4f}\n")

    with PdfPages(os.path.join(test_dir, "metrics_plots.pdf")) as pdf:
        make_metrics_plot(
            hit_pred_true_aris,
            hit_target_true_aris,
            hit_perfect_true_aris, 
            hit_baseline_pred_true_aris,
            xlabel="ARI"
        )
        pdf.savefig()
        plt.close()
        make_metrics_plot(
            hit_pred_true_purities,
            hit_target_true_purities,
            hit_perfect_true_purities, 
            hit_baseline_pred_true_purities,
            xlabel="Purity"
        )
        pdf.savefig()
        plt.close()
        make_metrics_plot(
            hit_pred_true_completenesses,
            hit_target_true_completenesses,
            hit_perfect_true_completenesses, 
            hit_baseline_pred_true_completenesses,
            xlabel="Completeness"
        )
        pdf.savefig()
        plt.close()

        if args.baseline_root_file is not None:
            make_metrics_plot(
                hit_pred_true_aris, [], [], hit_baseline_pred_true_aris, xlabel="ARI"
            )
            pdf.savefig()
            plt.close()
            make_metrics_plot(
                hit_pred_true_purities, [], [], hit_baseline_pred_true_purities,
                xlabel="Purity"
            )
            pdf.savefig()
            plt.close()
            make_metrics_plot(
                hit_pred_true_completenesses, [], [], hit_baseline_pred_true_completenesses,
                xlabel="Completeness"
            )
            pdf.savefig()
            plt.close()

def make_metrics_plot(metrics_pred, metrics_target, metrics_perfect, metrics_baseline, xlabel=""):
    fig, ax = plt.subplots(1, 1, figsize=(7,5))

    pred_hist, bins = np.histogram(metrics_pred, bins=100, range=(0,1), density=True)
    x = bins[:-1]
    ax.hist(x, bins=100, range=(0,1), weights=pred_hist, histtype="step", label="Pred")

    if len(metrics_target):
        target_hist, _ = np.histogram(metrics_target, bins=100, range=(0,1), density=True)
        ax.hist(x, bins=100, range=(0,1), weights=target_hist, histtype="step", label="Target")

    if len(metrics_perfect):
        perfect_hist, _ = np.histogram(metrics_perfect, bins=100, range=(0,1), density=True)
        ax.hist(
            x,
            bins=100, range=(0,1),
            weights=perfect_hist,
            histtype="step", label="Perfect", linestyle="dashed"
        )

    if len(metrics_baseline):
        baseline_hist, _ = np.histogram(metrics_baseline, bins=100, range=(0,1), density=True)
        ax.hist(x, bins=100, range=(0,1), weights=baseline_hist, histtype="step", label="Baseline")

    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel("Density", loc="top")
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [
        Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles
    ]
    ax.legend(new_handles, labels, fontsize=12, loc="upper center", ncol=2, borderpad=0.5)

    fig.tight_layout()

def get_mean(hit_metrics, n_hits_mask):
    hit_metrics = np.array(hit_metrics)
    return np.mean(hit_metrics[n_hits_mask])

def calc_metrics(y_true, y_pred, only_ari=False):
    ari = adjusted_rand_score(y_true, y_pred)
    if only_ari:
        return ari
    purity, completeness = _calc_purity_completeness(y_true, y_pred)
    return ari, purity, completeness

def _calc_purity_completeness(y_true, y_pred):
    C = contingency_matrix(y_true, y_pred)
    N = C.sum()

    # weighted by num hits
    purity = np.sum(np.max(C, axis=0)) / N
    completeness = np.sum(np.max(C, axis=1)) / N

    return purity, completeness

def cluster_from_similarity(t_sim, clusters, args):
    if args.clustering_mode == 1:
        return _connected_clustering(t_sim, clusters, args)
    elif args.clustering_mode == 2:
        return _agglomerative_clustering(t_sim, clusters, args)
    elif args.clustering_mode == 3:
        return _leiden_clustering(t_sim, clusters)
    elif args.clustering_mode == 4:
        return _affinity_clustering(t_sim, clusters)
    elif args.clustering_mode == 5:
        return _connected_accessory_clustering(t_sim, clusters, args)
    elif args.clustering_mode == 6:
        return _connected_accessory_clustering_2stage(t_sim, clusters, args)
    else:
        raise ValueError(f"clustering_mode {args.clustering_mode} invalid")

def _affinity_clustering(t_sim, clusters):
    cluster_labels = affinity_clustering(t_sim, clusters)
    hit_labels = [
        label for i_cluster, label in enumerate(cluster_labels) for _ in clusters[i_cluster].hits
    ]
    return np.array(hit_labels)

def _leiden_clustering(t_sim, clusters, resolution=1.0, min_weight=0.0):
    cluster_labels = leiden_clustering(
        t_sim, clusters, resolution=resolution, min_weight=min_weight
    )
    hit_labels = [
        label for i_cluster, label in enumerate(cluster_labels) for _ in clusters[i_cluster].hits
    ]
    return np.array(hit_labels)

def _agglomerative_clustering(t_sim, clusters, args):
    cluster_labels = agglomerative_clustering(t_sim, clusters, args.sim_threshold)
    hit_labels = [
        label for i_cluster, label in enumerate(cluster_labels) for _ in clusters[i_cluster].hits
    ]
    return np.array(hit_labels)

def _connected_accessory_clustering(t_sim, clusters, args):
    cluster_labels = connected_accessory_clustering(t_sim, clusters, args.sim_threshold)
    # -- XXX Ad-hoc visualisation
    # unmerged_acc_clusters = [ cluster for i, cluster in enumerate(clusters) if i in unmerged_acc_idxs ]
    # unmerged_acc_hits = [ hit for cluster in unmerged_acc_clusters for hit in cluster.hits ]
    # unmerged_acc_hit_labels = [ i for i, cluster in enumerate(unmerged_acc_clusters) for _ in cluster.hits ]
    # plot_clusters(
    #     unmerged_acc_hits, unmerged_acc_hit_labels,
    #     "/springbrook/share/physics/phsajw/dl_cluster_merging/plots/scratch/clustering_ex_7.pdf",
    #     "Leftover Accessory Cluster Edges",
    #     graph=G_acc, clusters=unmerged_acc_clusters
    # )
    # unmerged_acc_cluster_merged_labels = [ None for _ in range(len(clusters)) ]
    # for label, group in enumerate(acc_groups):
    #     for i in group:
    #         unmerged_acc_cluster_merged_labels[i] = label
    # unmerged_acc_hit_merged_labels = [ label for i, label in enumerate(unmerged_acc_cluster_merged_labels) for _ in clusters[i].hits if label is not None ]
    # plot_clusters(
    #     unmerged_acc_hits, unmerged_acc_hit_merged_labels,
    #     "/springbrook/share/physics/phsajw/dl_cluster_merging/plots/scratch/clustering_ex_8.pdf",
    #     "Leftover Accessory Clusters Merged"
    # )
    # -- XXX
    hit_labels = [
        label for i_cluster, label in enumerate(cluster_labels) for _ in clusters[i_cluster].hits
    ]
    # -- XXX Ad-hoc visualisation
    # plot_clusters(
    #     all_hits, hit_labels,
    #     "/springbrook/share/physics/phsajw/dl_cluster_merging/plots/scratch/clustering_ex_9.pdf",
    #     "Final Merged Clusters"
    # )
    # -- XXX
    return np.array(hit_labels)

def _connected_accessory_clustering_2stage(t_sim, clusters, args):
    cluster_labels = connected_accessory_clustering_2stage(
        t_sim, clusters, args.sim_threshold, args.sim_threshold_stage2
    )
    hit_labels = [
        label for i_cluster, label in enumerate(cluster_labels) for _ in clusters[i_cluster].hits
    ]
    return np.array(hit_labels)

def _connected_clustering(t_sim, clusters, args):
    cluster_labels = connected_clustering(t_sim, clusters, args.sim_threshold)
    hit_labels = [
        label for i_cluster, label in enumerate(cluster_labels) for _ in clusters[i_cluster].hits
    ]
    return np.array(hit_labels)

def get_true_clusterings(clusters):
    hit_labels_from_hit = [ int(hit.main_mc_id) for cluster in clusters for hit in cluster.hits ]
    return np.array(hit_labels_from_hit)

def get_perfect_clusterings(clusters):
    hit_labels_from_cluster = [
        int(cluster.main_mc_id) if cluster.main_mc_id is not None else -1
        for cluster in clusters
            for _ in cluster.hits
    ]
    return np.array(hit_labels_from_cluster)

def get_pred_target_sim(model, clusters, event, view, hit_feature_preset, collate_fn, conf):
    data = _make_input_data(clusters, event, view, hit_feature_preset, collate_fn, conf)
    model.set_input(data)
    model.test(compute_loss=False)
    return (
        model.get_current_tensors()["ev_t_sim"][0][0],
        model.get_current_tensors()["ev_t_sim_target"][0][0]
    )

def iterative_clustering(
    model, clusters, event, view, collate_fn, conf, args, use_target=False
):
    data = _make_input_data(
        clusters, event, view, args.hit_feature_preset, collate_fn, conf, make_mc_cnts=True
    )

    n_hits_fn = lambda t_cluster: t_cluster.size(0)
    clustering_fn = (
        lambda sim, clusters: connected_accessory_clustering(
            sim, clusters, args.sim_threshold, n_hits_fn
        )
    )

    cluster_groups = [ [ cluster ] for cluster in clusters ]
    aug_tier = 0
    while True:
        model.set_input(data)
        if use_target:
            t_sim = model.get_current_tensors()["ev_t_sim_target"][0].detach().cpu()
        else:
            model.test(compute_loss=True)
            t_sim = model.get_current_tensors()["ev_t_sim"][0].detach().cpu()

        ret = gen_single_aug(
            data["clusters"][0], data["mc_id_cnts"][0], t_sim, clustering_fn, aug_tier + 1, conf,
            ret_merges=True
        )
        if ret is None:
            break
        new_input, cluster_labels, cluster_labels_ordered = ret
        data = collate_fn([new_input])

        label_to_cluster_group = defaultdict(list)
        for label, cluster_group in zip(cluster_labels, cluster_groups):
            label_to_cluster_group[label] += cluster_group
        new_cluster_groups = []
        for label in cluster_labels_ordered:
            new_cluster_groups.append(label_to_cluster_group[label])
        cluster_groups = new_cluster_groups

        aug_tier += 1

    # Convert groups into correctly ordered label list
    assert len(clusters) == len({ cluster.id for cluster in clusters })
    id_to_idx = { cluster.id : i for i, cluster in enumerate(clusters) }
    cluster_labels = [ None for _ in range(len(clusters)) ]
    for label, cluster_group in enumerate(cluster_groups):
        for cluster in cluster_group:
            cluster_labels[id_to_idx[cluster.id]] = label
    
    hit_labels = [
        label for i_cluster, label in enumerate(cluster_labels) for _ in clusters[i_cluster].hits
    ]
    return np.array(hit_labels)

def _make_input_data(
    clusters, event, view, hit_feature_preset, collate_fn, conf, make_mc_cnts=False
):
    t_clusters = []
    for cluster in clusters:
        cluster_data = make_cluster_data(cluster, event, view, hit_feature_preset)
        t_clusters.append(torch.tensor(cluster_data, dtype=torch.float32))
    if conf.hit_feat_add_cardinality:
        t_clusters = add_cardinality_feature(t_clusters)
    if conf.hit_feat_add_aug_tier:
        t_clusters = add_aug_tier_feature(t_clusters, 0)
    for t in t_clusters:
        scale_cluster_tensor_inplace(
            t, conf.hit_feat_scaling_factors, conf.hit_feat_log_transforms
        )
    t_sim_target = get_similarity_matrix(clusters)
    event_data = { "input" : t_clusters, "target" : t_sim_target }

    if make_mc_cnts:
        mc_ids = [
            torch.tensor([ mc_id for mc_id in cluster.mc_id_cnt.keys() ], dtype=torch.long)
            for cluster in clusters
        ]
        mc_cnts = [
            torch.tensor([ mc_cnt for mc_cnt in cluster.mc_id_cnt.values() ], dtype=torch.long)
            for cluster in clusters
        ]
        mc_id_cnts = [
            Counter({ id : cnt for id, cnt in zip(ids.tolist(), cnts.tolist()) })
            for ids, cnts in zip(mc_ids, mc_cnts)
        ]
        event_data["mc_id_cnts"] = mc_id_cnts

    input_data = collate_fn.__call__([event_data])
    return input_data

def prep_model(conf, weights_file):
    model = ClusterMergeNet(conf)
    model.eval()
    model.load_networks(weights_file)
    if PRINT_N_PARAMS:
        model.print_num_params()
    if PRINT_FORWARD_PASS:
        model.print_forward_pass()

    return model

def prep_test_dir(test_dir):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    else:
        logger.warning(f"{test_dir} already exists, data may be overwritten")
    logger.info(f"Test dir will be {test_dir}")

def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file")
    parser.add_argument("weights_file")
    parser.add_argument("root_file")
    parser.add_argument("root_treename")

    parser.add_argument("--view", default="W", choices=["U", "V", "W"])
    parser.add_argument(
        "--hit_feature_preset", type=int, default=1, choices=range(1, 9),
        help=(
            "1 - Cartesian | "
            "2 - Cartesian w/ cheat | "
            "3 - Cartesian w/ summary | "
            "4 - Polar | "
            "5 - Polar w/ summary | "
            "6 - Cartesian w/ wire pitch | "
            "7 - Cartesian + Polar w/ wire pitch |"
            "8 - Cartesian + Polar w/ wire pitch + View one-hot"
        )
    )
    parser.add_argument(
        "--clustering_mode", type=int, default=1, choices=range(1, 8),
        help=(
            "1 - Connected components | "
            "2 - Agglomerative | "
            "3 - Leiden community | "
            "4 - Affinity | "
            "5 - Connected components, handle accessory clusters | "
            "6 - 5 w/ 2nd stage using reduced similarity threshold | "
            "7 - 5 applied iteratively"
        )
    )
    parser.add_argument(
        "--sim_threshold", type=float, default=0.5, 
        help=(
            "Threshold value for similarity. Use in clustering modes: "
            "1 - define adjacency matrix, "
            "2 - distance threshold, "
            "3 - nothing, "
            "4 - nothing, "
            "5 - define adjacency matrix and similarity threshold, "
            "6 - define stage 1 adjacency matrix and similarity threshold, "
            "7 - define adjacency matrix and similarity threshold"
        )
    )
    parser.add_argument(
        "--sim_threshold_stage2", type=float, default=0.5, 
        help=(
            "Second clustering stage threshold value for similarity. Use in clustering modes: "
            "1 - nothing, "
            "2 - nothing, "
            "3 - nothing, "
            "4 - nothing, "
            "5 - nothing, "
            "6 - apply at stage 2 clustering "
            "7 - nothing"
        )
    )
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--batch_mode", action="store_true")
    parser.add_argument("--test_dir_suffix", type=str, default="")
    parser.add_argument(
        "--baseline_root_file", type=str, default=None,
        help="Training ROOT file with clusters from baseline pandora clustering"
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    setup_logging()
    args = parse_cli()
    if args.batch_mode:
        tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=True)
    main(args)
