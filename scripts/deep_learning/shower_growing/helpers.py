import sys, math, itertools, os, random
from collections import defaultdict, Counter

import logging; logger = logging.getLogger("the_logger")

import numpy as np
import matplotlib; from matplotlib import pyplot as plt
import networkx as nx
import joblib

import torch

""" Start - plotting helpers """

def plot_pred_target(tensors, savedir, cntr, cntr_max, conf, unscale=True):
    ev_t_clusters = [ [ None for _ in range(t.size(-1)) ] for t in tensors["ev_t_sim_target"] ]
    for chunk_idx, ev_idxs in enumerate(tensors["chunked_t_cluster_ev_idxs"]):
        for row_idx, ev_idx in enumerate(ev_idxs):
            mask = tensors["chunked_t_clusters_mask"][chunk_idx]
            t_cluster = tensors["chunked_t_clusters"][chunk_idx][row_idx]
            if mask is not None:
                t_cluster = t_cluster[~(mask[row_idx])]
            ev_t_clusters[ev_idx[0]][ev_idx[1]] = t_cluster

    for i_ev, t_clusters in enumerate(ev_t_clusters):
        view = get_view(conf.plot_params["view"], t_clusters[0][0])

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        c_iter = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        for t_cluster in t_clusters:
            if unscale:
                unscale_cluster_tensor_inplace(
                    t_cluster, conf.hit_feat_scaling_factors, conf.hit_feat_log_transforms
                )
            c = next(c_iter)
            for t_hit in t_cluster:
                if conf.plot_params["has_summary_token"] and t_hit[-1].item() == 1.:
                    continue
                if conf.plot_params["polar_coords"]:
                    r = t_hit[0].item()
                    c_theta = t_hit[1].item()
                    s_theta = t_hit[2].item()
                    x = c_theta * r
                    z = s_theta * r
                    x_width = t_hit[conf.plot_params["x_width_idx"]].item()
                else:
                    x = t_hit[0].item()
                    z = t_hit[1].item()
                    x_width = t_hit[conf.plot_params["x_width_idx"]].item()
                z_width = get_pitch(view)
                patch_corner = (x - (x_width / 2), z - (z_width / 2))
                ax[0].add_patch(
                    matplotlib.patches.Rectangle(
                        patch_corner, x_width, z_width, fill=False, edgecolor=c, linewidth=0.3
                    )
                )
        ax[0].autoscale_view()
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("z")
        ax[0].set_title(f"Reco Clusters (view {view})")

        ax[1].imshow(
            tensors["ev_t_sim"][i_ev][0].float().detach().cpu().numpy(),
            vmin=0, vmax=1, cmap="Greys"
        )
        ax[1].set_title("Predicted Similarity Matrix")

        cax = ax[2].imshow(
            tensors["ev_t_sim_target"][i_ev][0].float().detach().cpu().numpy(),
            vmin=0, vmax=1, cmap="Greys"
        )
        cbar = fig.colorbar(cax, ax=ax[1:], fraction=0.046, pad=0.04)
        cbar.set_label("Similarity (0-1)")
        ax[2].set_title("Target Similarity Matrix")

        if savedir is not None:
            plt.savefig(os.path.join(savedir, f"{cntr}_val.pdf"))
            plt.close()

        if cntr is not None:
            cntr += 1
            if cntr > cntr_max:
                return cntr

    return cntr

def plot_clusterings(
    hits, initial_labels, pred_labels, true_labels, saveloc, conf,
    baseline_labels=None, target_labels=None
):
    random.seed(0)
    def random_rgb():
        return (random.random(), random.random(), random.random())

    def draw_hit(hit, ax, c, special=False, linewidth=0.3, alpha=1.0):
        x = hit.x
        z = hit.z
        x_width = hit.x_width
        z_width = get_pitch(conf.plot_params["view"])
        patch_corner = (x - (x_width / 2), z - (z_width / 2))
        if special:
            c = "r"
            alpha=0.2
            linewidth=0.6
        ax.add_patch(
            matplotlib.patches.Rectangle(
                patch_corner, x_width, z_width,
                fill=False, edgecolor=c, linewidth=linewidth, alpha=alpha
            )
        )

    if baseline_labels is None and target_labels is None:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    elif baseline_labels is None or target_labels is None:
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    else:
        fig, ax = plt.subplots(1, 5, figsize=(22, 4))

    cluster_size = defaultdict(int)
    for label in initial_labels:
        cluster_size[label] += 1
    main_hits = [ hit for hit, label in zip(hits, initial_labels) if cluster_size[label] >= 4 ]
    if not main_hits:
        main_hits = hits
    x_low, x_high = min(hit.x for hit in main_hits), max(hit.x for hit in main_hits)
    x_low, x_high = x_low - (x_high - x_low) * 0.05, x_high + (x_high - x_low) * 0.05
    z_low, z_high = min(hit.z for hit in main_hits), max(hit.z for hit in main_hits)
    z_low, z_high = z_low - (z_high - z_low) * 0.05, z_high + (z_high - z_low) * 0.05

    cs = { label : random_rgb() for label in set(initial_labels) }
    for hit, label in zip(hits, initial_labels):
        c = cs[label]
        draw_hit(hit, ax[0], c)
    ax[0].set_xlim(x_low, x_high)
    ax[0].set_ylim(z_low, z_high)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("z")
    ax[0].set_title(f"Initial Clusters ({len(cs)}) (view {conf.plot_params['view']})")

    cs = { label : random_rgb() for label in set(pred_labels) }
    for hit, label in zip(hits, pred_labels):
        c = cs[label]
        draw_hit(hit, ax[1], c)
    ax[1].set_xlim(x_low, x_high)
    ax[1].set_ylim(z_low, z_high)
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("z")
    ax[1].set_title(f"Pred Clusters ({len(cs)}) (view {conf.plot_params['view']})")


    cs = { label : random_rgb() for label in set(true_labels) }
    for hit, label in zip(hits, true_labels):
        c = cs[label]
        draw_hit(hit, ax[2], c, special=(label == -1))
    ax[2].set_xlim(x_low, x_high)
    ax[2].set_ylim(z_low, z_high)
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("z")
    ax[2].set_title(f"True Clusters ({len(set(cs) - {-1})}) (view {conf.plot_params['view']})")

    i_baseline = 3
    if target_labels is not None:
        cs = { label : random_rgb() for label in set(target_labels) }
        for hit, label in zip(hits, target_labels):
            c = cs[label]
            draw_hit(hit, ax[3], c)
        ax[3].set_xlim(x_low, x_high)
        ax[3].set_ylim(z_low, z_high)
        ax[3].set_xlabel("x")
        ax[3].set_ylabel("z")
        ax[3].set_title(f"Target Clusters ({len(cs)}) (view {conf.plot_params['view']})")
        i_baseline += 1

    if baseline_labels is not None:
        cs = { label : random_rgb() for label in set(baseline_labels) }
        for hit, label in zip(hits, baseline_labels):
            c = cs[label]
            draw_hit(hit, ax[i_baseline], c)
        ax[i_baseline].set_xlim(x_low, x_high)
        ax[i_baseline].set_ylim(z_low, z_high)
        ax[i_baseline].set_xlabel("x")
        ax[i_baseline].set_ylabel("z")
        ax[i_baseline].set_title(
            f"Baseline Reco Clusters ({len(cs)}) (view {conf.plot_params['view']})"
        )

    fig.tight_layout()

    if saveloc is not None:
        plt.savefig(saveloc)
        plt.close()

def plot_pred_target_adj(t_sim_pred, t_adj_pred, t_sim_target, t_adj_target, assoc_thres, saveloc):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))

    t_sim_pred = t_sim_pred.float().detach().cpu().numpy()
    t_sim_target = t_sim_target.float().detach().cpu().numpy()
    t_adj_pred = t_adj_pred.float().detach().cpu().numpy()
    t_adj_target = t_adj_target.float().detach().cpu().numpy()

    cax_l = ax[0, 0].imshow(t_sim_pred, vmin=0, vmax=1, cmap="Greys")
    ax[0, 0].set_title("Pred. Sim. Matrix")

    ax[0, 1].imshow(t_sim_target, vmin=0, vmax=1, cmap="Greys")
    ax[0, 1].set_title("Target Sim. Matrix")

    cax_r = ax[0, 2].imshow((t_sim_pred - t_sim_target), vmin=-1, vmax=1, cmap="seismic")
    ax[0, 2].set_title("(Pred. - Target) Sim. Matrix")

    fig.colorbar(cax_l, ax=ax[:, :], fraction=0.046, pad=0.05, location="left")
    fig.colorbar(cax_r, ax=ax[:, :], fraction=0.046, pad=0.05, location="right")

    ax[1, 0].imshow(t_adj_pred, vmin=0, vmax=1, cmap="Greys")
    ax[1, 0].set_title(f"Pred. Adj. Matrix (Sim. > {assoc_thres})")

    ax[1, 1].imshow(t_adj_target, vmin=0, vmax=1, cmap="Greys")
    ax[1, 1].set_title(f"Target Adj. Matrix (Sim. > {assoc_thres})")

    ax[1, 2].imshow((t_adj_pred - t_adj_target), vmin=-1, vmax=1, cmap="seismic"
    )
    ax[1, 2].set_title(f"(Pred. - Target) Adj. Matrix (Sim. > {assoc_thres})")

    if saveloc is not None:
        plt.savefig(saveloc)
        plt.close()

def plot_loss_weights(weights, saveloc):
    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    weights = weights.float().detach().cpu().numpy()
    cax = ax.imshow(
        weights,
        norm=matplotlib.colors.LogNorm(vmin=weights.min()+1e-12, vmax=weights.max()), cmap="Greys"
    )
    ax.set_title("Loss Weights (clusters ordered by num. hits)")
    fig.colorbar(cax, ax=ax)
    plt.savefig(saveloc)
    plt.close()

def plot_clusters(hits, hit_labels, saveloc, title="", graph=None, clusters=None):
    random.seed(0)
    def random_rgb():
        return (random.random(), random.random(), random.random())

    def draw_hit(hit, ax, c, special=False, linewidth=0.3, alpha=1.0):
        x = hit.x
        z = hit.z
        x_width = hit.x_width
        z_width = get_pitch(6)
        patch_corner = (x - (x_width / 2), z - (z_width / 2))
        if special:
            c = "r"
            alpha=0.2
            linewidth=0.6
        ax.add_patch(
            matplotlib.patches.Rectangle(
                patch_corner, x_width, z_width,
                fill=False, edgecolor=c, linewidth=linewidth, alpha=alpha
            )
        )

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    cs = { label : random_rgb() for label in set(hit_labels) }
    for hit, label in zip(hits, hit_labels):
        c = cs[label]
        draw_hit(hit, ax, c)
    ax.autoscale_view()
    ax.set_title(title, fontsize=16)

    if graph is not None:
        graph.remove_edges_from(nx.selfloop_edges(graph))
        nx.draw_networkx(
            graph,
            pos={ i : (cluster.hits[0].x, cluster.hits[0].z) for i, cluster in enumerate(clusters) },
            with_labels=False, ax=ax, node_size=1, width=0.2
        )
    ax.set_ylim(-20, 360)
    ax.set_xlim(-70, 50)
    ax.set_axis_off()

    fig.tight_layout()
    plt.savefig(saveloc)
    plt.close()

def make_2d_plot(
    arr_x, arr_y, xlabel, ylabel, n_bins, range_bins, savepath,
    draw_identity=False, cbar_label=None, normalise_cols=False
):
    hist2d, bins_x, bins_y = np.histogram2d(arr_x, arr_y, bins=n_bins, range=range_bins)
    if normalise_cols:
        hist2d = hist2d.T
        col_sums = hist2d.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        hist2d /= col_sums
        hist2d = hist2d.T

    fig, ax = plt.subplots(1, 1, layout="compressed", figsize=(7, 6))
    # extent = [bins_y[0], bins_y[-1], bins_x[0], bins_x[-1]]
    extent = [bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]]
    im = ax.imshow(
        np.ma.masked_where(hist2d == 0, hist2d).T,
        origin="lower", interpolation="none", extent=extent, cmap="viridis", aspect="auto"
    )
    if draw_identity:
        min_val, max_val = min(bins_x[0], bins_y[0]), max(bins_x[-1], bins_y[-1])
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", location="right")
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    plt.savefig(savepath)
    plt.close()

def make_1d_plot(arr, xlabel, ylabel, n_bins, range_bins, savepath):
    fig, ax = plt.subplots(1, 1, layout="compressed", figsize=(7, 6))
    ax.hist(arr, bins=n_bins, range=range_bins, histtype="step")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(savepath)
    plt.close()

""" End - plotting helpers """

""" Start - feature helpers """

SCALING_FACTORS = { # From pandora's 1x2x6 FD HD detector boundaries
    "polar_r" : (
        1 / math.sqrt((362.622 - -362.622)**2 + (1393.46 - -0.876221)**2 + (603.924 - -603.924)**2)
    ),
    "cartesian_x" : 1 / (362.622 - -362.622),
    "cartesian_z" : 1 / (1393.46 - -0.876221)
}

def scale_cluster_tensor_inplace(t, hit_scaling_factors, hit_log_transforms):
    for idx, scaling_type in hit_scaling_factors.items():
        t[:, idx] *= SCALING_FACTORS[scaling_type]
    for idx in hit_log_transforms:
        t[:, idx] = torch.log(t[:, idx])

def unscale_cluster_tensor_inplace(t, hit_scaling_factors, hit_log_transforms):
    for idx, scaling_type in hit_scaling_factors.items():
        t[:, idx] /= SCALING_FACTORS[scaling_type]
    for idx in hit_log_transforms:
        t[:, idx] = torch.exp(t[:, idx])

def add_cardinality_feature(t_cluster_lst, active=True):
    return [
        torch.cat(
            [
                t,
                torch.tensor(
                    [t.size(0), len(t_cluster_lst)] if active else (torch.rand(2) + 1.).tolist(),
                    dtype=t.dtype, device=t.device
                ).repeat(t.size(0), 1)
            ],
            dim=1
        )
        for t in t_cluster_lst
    ]

def update_cardinality_feature(
    t_cluster_lst, hit_scaling_factors, hit_log_transforms, n_clusters_idx, n_hits_idx
):
    device, dtype = t_cluster_lst[0].device, t_cluster_lst[0].dtype

    n_clusters = torch.tensor(len(t_cluster_lst), device=device, dtype=dtype)
    if n_clusters_idx in hit_scaling_factors:
        n_clusters *= SCALING_FACTORS[hit_scaling_factors[n_clusters_idx]]
    if n_clusters_idx in hit_log_transforms:
        n_clusters = torch.log(n_clusters)

    for t in t_cluster_lst:
        n_hits = torch.tensor(t.size(0), device=device, dtype=dtype)
        if n_hits_idx in hit_scaling_factors:
            n_hits *= SCALING_FACTORS[hit_scaling_factors[n_hits_idx]]
        if n_hits_idx in hit_log_transforms:
            n_hits = torch.log(n_hits)

        t[:, n_clusters_idx] = n_clusters
        t[:, n_hits_idx] = n_hits

def add_aug_tier_feature(t_cluster_lst, aug_tier, active=True):
    return [
        torch.cat(
            [
                t,
                torch.tensor(
                    [aug_tier + 1] if active else (torch.rand(1) + 1.).tolist(),
                    dtype=t.dtype, device=t.device
                ).repeat(t.size(0), 1)
            ],
            dim=1
        )
        for t in t_cluster_lst
    ]

def update_aug_tier_feature(
    t_cluster_lst, hit_scaling_factors, hit_log_transforms, aug_tier, aug_tier_idx
):
    device, dtype = t_cluster_lst[0].device, t_cluster_lst[0].dtype

    aug_tier = torch.tensor(aug_tier + 1, device=device, dtype=dtype)
    if aug_tier_idx in hit_scaling_factors:
        aug_tier *= SCALING_FACTORS[hit_scaling_factors[aug_tier_idx]]
    if aug_tier_idx in hit_log_transforms:
        aug_tier = torch.log(aug_tier)

    for t in t_cluster_lst:
        t[:, aug_tier_idx] = aug_tier

def get_added_hit_feat_idxs(hit_feat_vec_dim, add_cardinality, add_aug_tier):
    if add_cardinality and add_aug_tier:
        return hit_feat_vec_dim - 3, hit_feat_vec_dim - 2, hit_feat_vec_dim - 1
    if add_cardinality and not add_aug_tier:
        return hit_feat_vec_dim - 2, hit_feat_vec_dim - 1, None
    if not add_cardinality and add_aug_tier:
        return None, None, hit_feat_vec_dim - 1
    return None, None, None

""" End - feature helpers """

""" Start - Iterative augmentation helpers """

def gen_augs(data, pred_sims, clustering_fn, collate_fn, max_n_jobs, aug_tier, conf, dataset=None):
    if max_n_jobs: # This seems to make things slighty slower...
        new_inputs = joblib.Parallel(n_jobs=min(max_n_jobs, len(data["clusters"])))(
            joblib.delayed(gen_single_aug)(
                clusters, mc_id_cnts, pred_sim.detach().cpu(), clustering_fn, aug_tier, conf,
                dataset=dataset
            )
            for clusters, mc_id_cnts, pred_sim in zip(
                data["clusters"], data["mc_id_cnts"], pred_sims
            )
        )
    else:
        new_inputs = [
            gen_single_aug(
                clusters, mc_id_cnts, pred_sim.detach().cpu(), clustering_fn, aug_tier, conf,
                dataset=dataset
            )
            for clusters, mc_id_cnts, pred_sim in zip(
                data["clusters"], data["mc_id_cnts"], pred_sims
            )
        ]

    new_inputs = [ el for el in new_inputs if el is not None ]
    if not len(new_inputs):
        return None

    return collate_fn(new_inputs)

def gen_single_aug(
    clusters, mc_id_cnts, pred_sim, clustering_fn, aug_tier, conf, ret_merges=False, dataset=None
):
    cluster_labels = clustering_fn(pred_sim[0], clusters)

    if len(set(cluster_labels)) == len(cluster_labels):
        return None

    cluster_groups, mc_id_cnt_groups, uniq_mc_ids = defaultdict(list), defaultdict(list), set()
    for label, cluster, mc_id_cnt in zip(cluster_labels, clusters, mc_id_cnts):
        cluster_groups[label].append(cluster)
        mc_id_cnt_groups[label].append(mc_id_cnt)
        uniq_mc_ids = uniq_mc_ids | set(mc_id_cnt)
    uniq_labels = list(cluster_groups.keys())
    new_clusters = [ torch.cat(cluster_groups[label], dim=0) for label in uniq_labels ]
    new_mc_id_cnts = [ sum(mc_id_cnt_groups[label], Counter()) for label in uniq_labels ]

    if dataset is None or dataset.hit_feat_add_on:
        hit_feat_data = dataset if dataset is not None else conf
        n_hits_idx, n_clusters_idx, aug_tier_idx = get_added_hit_feat_idxs(
            hit_feat_data.hit_feat_vec_dim,
            hit_feat_data.hit_feat_add_cardinality,
            hit_feat_data.hit_feat_add_aug_tier
        )
        if hit_feat_data.hit_feat_add_cardinality:
            update_cardinality_feature(
                new_clusters,
                hit_feat_data.hit_feat_scaling_factors,
                hit_feat_data.hit_feat_log_transforms,
                n_hits_idx,
                n_clusters_idx
            )
        if hit_feat_data.hit_feat_add_aug_tier:
            update_aug_tier_feature(
                new_clusters,
                hit_feat_data.hit_feat_scaling_factors,
                hit_feat_data.hit_feat_log_transforms,
                aug_tier,
                aug_tier_idx
            )

    # Old and slow :(
    # new_sim = torch.zeros(len(new_clusters), len(new_clusters), dtype=torch.float32)
    # for i_cluster_a, (cluster_a, mc_id_cnts_a) in enumerate(zip(new_clusters, new_mc_id_cnts)):
    #     for i_cluster_b, (cluster_b, mc_id_cnts_b) in enumerate(zip(new_clusters, new_mc_id_cnts)):
    #         sim = 0
    #         for mc_id, mc_cnt_a in mc_id_cnts_a.items():
    #             if mc_id == -1 or mc_id not in mc_id_cnts_b:
    #                 continue
    #             sim += (mc_cnt_a / cluster_a.size(0)) * (mc_id_cnts_b[mc_id] / cluster_b.size(0))
    #         new_sim[i_cluster_a][i_cluster_b] = sim

    t_mc_cnt = torch.zeros(len(new_clusters), len(uniq_mc_ids), dtype=torch.float32)
    mc_id_to_idx = { mc_id : idx for mc_id, idx in zip(uniq_mc_ids, range(len(uniq_mc_ids))) }
    for i, cnts in enumerate(new_mc_id_cnts):
        for mc_id, cnt in cnts.items():
            if mc_id != -1:
                t_mc_cnt[i, mc_id_to_idx[mc_id]] = cnt
    t_cluster_sizes = torch.tensor(
        [ t_cluster.size(0) for t_cluster in new_clusters ], dtype=torch.float32
    )
    t_mc_purities = t_mc_cnt / t_cluster_sizes[:, None]
    new_sim = t_mc_purities @ t_mc_purities.T

    new_input = { "input" : new_clusters, "target" : new_sim, "mc_id_cnts" : new_mc_id_cnts }

    if ret_merges:
        return new_input, cluster_labels, uniq_labels
    return new_input

""" End - Iterative augmentation helpers """

""" Start - misc helpers """

def get_view(view, t_hit):
    if view < 0: 
        if view == -1:
            lo, up = -3, None 
        else:
            up = view + 1
            lo = up - 3
        if t_hit[lo:up].bool().tolist() == [True, False, False]:
            return 4
        if t_hit[lo:up].bool().tolist() == [False, True, False]:
            return 5
        if t_hit[lo:up].bool().tolist() == [False, False, True]:
            return 6
        raise ValueError(f"Failed to infer view from hit tensor {t_hit}")
    return view

def get_pitch(view):
    if view == 4 or 5:
        return 0.4667
    if view == 6:
        return 0.479
    raise ValueError(f"Invalid view: {view}")

def get_view_str(view):
    if view == 4:
        return "U"
    if view == 5:
        return "V"
    if view == 6:
        return "W"
    raise ValueError(f"Invalid view: {view}")

def setup_logging():
    cyan="\033[0;36m"#]
    yellow="\033[0;33m"#]
    colour_reset="\033[0m"#]

    # only handle one level... too much work for pretty colours :(
    class MyFilter(logging.Filter):
        def __init__(self, level):
            super().__init__()
            self.level = level
        def filter(self, record):
            return record.levelno == self.level

    def get_handler(level, colour):
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(level)
        f = logging.Formatter(f"{colour}[%(asctime)s - %(levelname)s - %(module)s]{colour_reset} %(message)s")
        h.setFormatter(f)
        h.addFilter(MyFilter(level))
        return h

    logger.addHandler(get_handler(logging.INFO, cyan))
    logger.addHandler(get_handler(logging.WARNING, yellow))
    logger.setLevel(logging.INFO)

def get_gpu_usage(device):
    tot_mem = torch.cuda.max_memory_allocated(device=device)
    return tot_mem / 1e9

""" End - misc helpers """
