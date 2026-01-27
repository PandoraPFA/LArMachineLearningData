import argparse, os, time, functools

import logging; logger = logging.getLogger("the_logger")

import uproot
from tqdm import tqdm
# from memory_profiler import memory_usage
import numpy as np
from matplotlib import pyplot as plt

import torch

from config_parser import get_config
from helpers import setup_logging, scale_cluster_tensor_inplace, make_2d_plot
from model import ClusterMergeNet
from data.read_cluster import read_events, make_cluster_data, get_similarity_matrix
from dataset import CollateClusters

def main(args):
    conf_overwrite_dict = { "device" : "cpu" }
    if args.use_looped_similarity_forward:
        conf_overwrite_dict[("net_inter_cluster_sim_params", "use_loop_implementation")] = True
    elif args.use_chunked_similarity_forward:
        conf_overwrite_dict[("net_inter_cluster_sim_params", "use_chunked_implementation")] = True
        conf_overwrite_dict[("net_inter_cluster_sim_params", "chunk_size")] = 1024
    conf = get_config(args.config_file, test=True, overwrite_dict=conf_overwrite_dict)

    model = prep_model(conf, args.weights_file)

    collate_fn = CollateClusters(conf)

    test_dir = os.path.join(os.path.dirname(args.weights_file), "torchscripts")
    prep_test_dir(test_dir)

    with torch.no_grad(): # Just in case
        tscript_net_intra_cluster_encoder = torch.jit.script(model.net_intra_cluster_encoder)
        tscript_net_inter_cluster_attn = torch.jit.script(model.net_inter_cluster_attn)
        tscript_net_inter_cluster_sim = torch.jit.script(model.net_inter_cluster_sim)

    tscript_net_intra_cluster_encoder.save(os.path.join(test_dir, "net_intra_cluster_encoder.pt"))
    tscript_net_inter_cluster_attn.save(os.path.join(test_dir, "net_inter_cluster_attn.pt"))
    tscript_net_inter_cluster_sim.save(os.path.join(test_dir, "net_inter_cluster_sim.pt"))

    if args.validation_root_file is None:
        return

    cluster_lens, hit_lens = [], []
    encoder_times, attn_times, sim_times = [], [], []
    # encoder_peak_mems, attn_peak_mems, sim_peak_mems = [], [], [] # These are not very useful
    py_tscript_pair_diffs = []

    events = read_events(
        uproot.open(args.validation_root_file)[args.validation_treename], n_events=10_000
    )
    for event in tqdm(events, desc="Processing test events..."):
        for view, clusters in event.view_clusters.items():
            for cluster in clusters:
                cluster.calc_main_mc()

            cluster_lens.append(len(clusters))
            hit_lens.append(sum(cluster.get_n_hits() for cluster in clusters))

            t_clusters, t_sim_target = get_test_data(
                clusters, event, view, args.validation_hit_preset, conf
            )

            t_sim_py = predict_py(t_clusters, t_sim_target, collate_fn, model)
            (
                t_sim_tscript,
                encoder_time, _,
                attn_time, _,
                sim_time, _
            ) = predict_tscript(
                t_clusters,
                tscript_net_intra_cluster_encoder,
                tscript_net_inter_cluster_attn,
                tscript_net_inter_cluster_sim
            )
            encoder_times.append(encoder_time * 1000)
            attn_times.append(attn_time * 1000)
            sim_times.append(sim_time * 1000)

            for diff in (t_sim_py - t_sim_tscript).flatten():
                py_tscript_pair_diffs.append(diff.item())

    cluster_lens = np.array(cluster_lens)
    hit_lens = np.array(hit_lens)
    encoder_times = np.array(encoder_times)
    attn_times = np.array(attn_times)
    sim_times = np.array(sim_times)
    py_tscript_pair_diffs = np.array(py_tscript_pair_diffs)

    total_times = encoder_times + attn_times + sim_times

    print(f"np.max(hit_lens)={np.max(hit_lens)}")
    print(f"np.max(cluster_lens)={np.max(cluster_lens)}")
    print(f"np.max(total_times)={np.max(total_times)}")

    n_bins, range_bins = 1000, (-0.0001, 0.0001)
    py_tscript_pair_diffs_hist, bins = np.histogram(
        py_tscript_pair_diffs, bins=n_bins, range=range_bins
    )
    x = bins[:-1]
    _, ax = plt.subplots(1, 1, layout="compressed", figsize=(7,5))
    ax.hist(x, bins=n_bins, range=range_bins, weights=py_tscript_pair_diffs_hist, histtype="step")
    ax.set_xlabel("(Pytorch - Torschript)")
    ax.set_ylabel("No. Cluster Pairs")
    plt.savefig(os.path.join(test_dir, "py_tscript_diffs.pdf"))
    plt.close()

    make_2d_plot(
        hit_lens, encoder_times, "Total No. Hits", "Cluster Encoder Inference Time (ms)",
        (50, 50), ((0, 10000), (0, 2000)),
        os.path.join(test_dir, "hits-encoder_times_hist2d.pdf")
    )
    make_2d_plot(
        hit_lens, attn_times, "Total No. Hits", "Event Attention Inference Time (ms)",
        (50, 50), ((0, 10000), (0, 2000)),
        os.path.join(test_dir, "hits-attn_times_hist2d.pdf")
    )
    make_2d_plot(
        hit_lens, sim_times, "Total No. Hits", "Pairwise Similarity Inference Time (ms)",
        (50, 50), ((0, 10000), (0, 2000)),
        os.path.join(test_dir, "hits-sim_times_hist2d.pdf")
    )
    make_2d_plot(
        hit_lens, sim_times, "Total No. Hits", "Pairwise Similarity Inference Time (ms)",
        (50, 50), ((0, 10000), (0, 50_000)),
        os.path.join(test_dir, "hits-sim_times_largerrange_hist2d.pdf")
    )
    make_2d_plot(
        hit_lens, total_times, "Total No. Hits", "Total Inference Time (ms)",
        (50, 50), ((0, 10000), (0, 6000)),
        os.path.join(test_dir, "hits-total_times_hist2d.pdf")
    )
    make_2d_plot(
        hit_lens, total_times, "Total No. Hits", "Total Inference Time (ms)",
        (50, 50), ((0, 10000), (0, 56_000)),
        os.path.join(test_dir, "hits-total_times_largerrange_hist2d.pdf")
    )
    make_2d_plot(
        cluster_lens, encoder_times, "Total No. Clusters", "Cluster Encoder Inference Time (ms)",
        (50, 50), ((0, 2000), (0, 2000)),
        os.path.join(test_dir, "clusters-encoder_times_hist2d.pdf")
    )
    make_2d_plot(
        cluster_lens, attn_times, "Total No. Clusters", "Event Attention Inference Time (ms)",
        (50, 50), ((0, 2000), (0, 2000)),
        os.path.join(test_dir, "clusters-attn_times_hist2d.pdf")
    )
    make_2d_plot(
        cluster_lens, sim_times, "Total No. Clusters", "Pairwise Similarity Inference Time (ms)",
        (50, 50), ((0, 2000), (0, 2000)),
        os.path.join(test_dir, "clusters-sim_times_hist2d.pdf")
    )
    make_2d_plot(
        cluster_lens, sim_times, "Total No. Clusters", "Pairwise Similarity Inference Time (ms)",
        (50, 50), ((0, 2000), (0, 50_000)),
        os.path.join(test_dir, "clusters-sim_times_largerrange_hist2d.pdf")
    )
    make_2d_plot(
        cluster_lens, total_times, "Total No. Clusters", "Total Inference Time (ms)",
        (50, 50), ((0, 2000), (0, 6000)),
        os.path.join(test_dir, "clusters-total_times_hist2d.pdf")
    )
    make_2d_plot(
        cluster_lens, total_times, "Total No. Clusters", "Total Inference Time (ms)",
        (50, 50), ((0, 2000), (0, 56_000)),
        os.path.join(test_dir, "clusters-total_times_largerrange_hist2d.pdf")
    )

def predict_tscript(
    t_clusters,
    tscript_net_intra_cluster_encoder,
    tscript_net_inter_cluster_attn,
    tscript_net_inter_cluster_sim
):
    def predict_encs(t_clusters):
        t_cluster_encs = []
        for t_cluster in t_clusters:
            # Checking if the ordering matters (it should not, *set* transformer!)
            perm = torch.randperm(t_cluster.size(0))
            t_cluster = t_cluster[perm]
            t_cluster_encs.append(tscript_net_intra_cluster_encoder(t_cluster.unsqueeze(0)))
        return t_cluster_encs
    t0 = time.perf_counter()
    # mem_usages, t_cluster_encs = memory_usage(
    #     (predict_encs, (t_clusters,)), interval=0.01, retval=True
    # )
    t_cluster_encs = predict_encs(t_clusters)
    enc_time = time.perf_counter() - t0
    # enc_peak_mem = max(mem_usages)

    def predict_attn(t_ev_cluster_enc):
        t_ev_cluster_attn = tscript_net_inter_cluster_attn(t_ev_cluster_enc)
        return t_ev_cluster_attn
    perm = torch.randperm(len(t_cluster_encs)) # Checking ordering here too
    inv_perm = torch.argsort(perm)
    t_cluster_encs = [ t_cluster_encs[i] for i in perm.tolist() ]
    t_ev_cluster_enc = torch.cat(t_cluster_encs, dim=1)
    t0 = time.perf_counter()
    # mem_usages, t_ev_cluster_attn = memory_usage(
    #     (predict_attn, (t_ev_cluster_enc,)), interval=0.01, retval=True
    # )
    t_ev_cluster_attn = predict_attn(t_ev_cluster_enc)
    attn_time = time.perf_counter() - t0
    # attn_peak_mem = max(mem_usages)

    def predict_sim(t_ev_cluster_attn):
        t_ev_sim = tscript_net_inter_cluster_sim(t_ev_cluster_attn)
        return t_ev_sim
    t0 = time.perf_counter()
    # mem_usages, t_ev_sim = memory_usage(
    #     (predict_sim, (t_ev_cluster_attn,)), interval=0.01, retval=True
    # )
    t_ev_sim = predict_sim(t_ev_cluster_attn)
    sim_time = time.perf_counter() - t0
    # sim_peak_mem = max(mem_usages)

    # Need to undo this shuffle to align with python prediction
    t_ev_sim = t_ev_sim[:, inv_perm][:, :, inv_perm]

    # return t_ev_sim[0], enc_time, enc_peak_mem, attn_time, attn_peak_mem, sim_time, sim_peak_mem
    return t_ev_sim[0], enc_time, 0., attn_time, 0., sim_time, 0.

def predict_py(t_clusters, t_sim_target, collate_fn, model):
    event_data = { "input" : t_clusters, "target" : t_sim_target }
    input_data = collate_fn.__call__([event_data])
    model.set_input(input_data)
    model.test(compute_loss=False)
    return model.get_current_tensors()["ev_t_sim"][0][0]

def get_test_data(clusters, event, view, hit_feature_preset, conf):
    t_clusters = []
    for cluster in clusters:
        cluster_data = make_cluster_data(cluster, event, view, hit_feature_preset)
        t_clusters.append(torch.tensor(cluster_data, dtype=torch.float32))
    t_sim_target = get_similarity_matrix(clusters)
    for t in t_clusters:
        scale_cluster_tensor_inplace(
            t, conf.hit_feat_scaling_factors, conf.hit_feat_log_transforms
        )
    return t_clusters, t_sim_target

def prep_test_dir(test_dir):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    else:
        logger.warning(f"{test_dir} already exists, data may be overwritten")

def prep_model(conf, weights_file):
    model = ClusterMergeNet(conf)
    model.eval()
    model.load_networks(weights_file)
    for net in model.nets:
        for ps in net.parameters():
            ps.requires_grad = False  # Just to be safe
    return model

def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file")
    parser.add_argument("weights_file")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_looped_similarity_forward", action="store_true")
    group.add_argument("--use_chunked_similarity_forward", action="store_true")

    parser.add_argument("--validation_root_file", type=str, default=None)
    parser.add_argument("--validation_treename", type=str, default=None)
    parser.add_argument("--validation_hit_preset", type=int, default=None, choices=range(1, 9))

    parser.add_argument("--batch_mode", action="store_true")

    args = parser.parse_args()

    if (
        args.validation_root_file is not None and
        (args.validation_treename is None or args.validation_hit_preset is None)
    ):
        raise argparse.ArgumentError(
            "--validation_treename and --validation_hit_preset required with --validation_root_file"
        )

    return args

if __name__ == "__main__":
    setup_logging()
    args = parse_cli()
    if args.batch_mode:
        tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=True)
    main(args)
