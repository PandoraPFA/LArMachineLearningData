"""
Read in training data - reco clusters written out to a TTree from a pandora alg
"""

import argparse, os

import numpy as np
from tqdm import tqdm; from tqdm_joblib import tqdm_joblib
import torch
import uproot
import joblib

import sys; sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers import get_pitch
from data.event import Event

def get_similarity_matrix(clusters):
    similarity_mat = torch.zeros(len(clusters), len(clusters), dtype=torch.float32)
    for i_cluster_a, cluster_a in enumerate(clusters):
        for i_cluster_b, cluster_b in enumerate(clusters):
            similarity = 0
            for mc_id, mc_cnt_a in cluster_a.mc_id_cnt.items(): # Dot product of purities wrt to MC particle "dimension"
                if mc_id == -1 or mc_id not in cluster_b.mc_id_cnt:
                    continue
                similarity += (
                    (mc_cnt_a / len(cluster_a.hits)) *
                    (cluster_b.mc_id_cnt[mc_id] / len(cluster_b.hits))
                )
            similarity_mat[i_cluster_a][i_cluster_b] = similarity

    # torch.set_printoptions(profile="full"); print(similarity_mat); torch.set_printoptions(profile="default")
    # for row in similarity_mat[:30]:
    #     print("")
    #     for col in row[:30]:
    #         print(f"{col:.2f}, ", end="")

    return similarity_mat

def make_cluster_data(cluster, event, view, preset):
    cluster_data = []

    if preset == 1: # Cartesian
        for hit in cluster.hits:
            hit_feat_vec = [
                hit.x, hit.z, hit.x_width, hit.x_gap_dist, hit.energy
            ]
            cluster_data.append(hit_feat_vec)

    elif preset == 2: # Cartesian w/ cheated feature
        for hit in cluster.hits:
            hit_feat_vec = [
                hit.x, hit.z, hit.x_width, hit.x_gap_dist, hit.energy, hit.main_mc_id
            ]
            cluster_data.append(hit_feat_vec)

    elif preset == 3: # Cartesian w/ summary token
        for hit in cluster.hits:
            hit_feat_vec = [
                hit.x, hit.z, hit.x_width, hit.x_gap_dist, hit.energy, 0.
            ]
            cluster_data.append(hit_feat_vec)
        summary_feat_vec = [
            float(cluster.get_n_hits()),
            float(event.get_n_hits(view)),
            float(event.get_n_clusters(view)),
            0., 0., 1.
        ]
        assert len(hit_feat_vec) == len(summary_feat_vec), "DUHHH DUHH"
        cluster_data.append(summary_feat_vec)

    elif preset == 4: # Polar
        for hit in cluster.hits:
            hit_feat_vec = [
                hit.r, hit.c_theta, hit.s_theta, hit.x_width, hit.x_gap_dist, hit.energy
            ]
            cluster_data.append(hit_feat_vec)

    elif preset == 5: # Polar w/ summary token
        for hit in cluster.hits:
            hit_feat_vec = [
                hit.r, hit.c_theta, hit.s_theta, hit.x_width, hit.x_gap_dist, hit.energy,
                0., 0.
            ]
            cluster_data.append(hit_feat_vec)
        summary_feat_vec = [
            float(cluster.get_n_hits()),
            float(event.get_n_hits(view)),
            float(event.get_n_clusters(view)),
            0., 0., 0., 0., 1.
        ]
        assert len(hit_feat_vec) == len(summary_feat_vec), "DUHHH DUHH"
        cluster_data.append(summary_feat_vec)

    elif preset == 6: # Cartesian w/ wire pitch
        for hit in cluster.hits:
            hit_feat_vec = [
                hit.x, hit.z, hit.x_width, get_pitch(view), hit.x_gap_dist, hit.energy
            ]
            cluster_data.append(hit_feat_vec)

    elif preset == 7: # Cartesian + polar w/ wire pitch
        for hit in cluster.hits:
            hit_feat_vec = [
                hit.r, hit.c_theta, hit.s_theta,
                hit.x, hit.z, hit.x_width,
                get_pitch(view), hit.x_gap_dist, hit.energy
            ]
            cluster_data.append(hit_feat_vec)

    elif preset == 8: # Cartesian + polar w/ wire pitch + view one-hot
        for hit in cluster.hits:
            hit_feat_vec = [
                hit.r, hit.c_theta, hit.s_theta,
                hit.x, hit.z, hit.x_width,
                get_pitch(view), hit.x_gap_dist, hit.energy
            ]
            if view == 4:
                hit_feat_vec += [1., 0., 0.]
            elif view == 5:
                hit_feat_vec += [0., 1., 0.]
            else:
                hit_feat_vec += [0., 0., 1.]
            cluster_data.append(hit_feat_vec)

    else:
        raise ValueError(f"preset {preset} not valid")

    return cluster_data

def process_event(i_event, event, args):
    for view, clusters in event.view_clusters.items():
        if view == 4:
            if args.out_dir_U is None:
                continue
            out_dir = args.out_dir_U
            suffix = "U"
        elif view == 5:
            if args.out_dir_V is None:
                continue
            out_dir = args.out_dir_V
            suffix = "V"
        elif view == 6:
            if args.out_dir_W is None:
                continue
            out_dir = args.out_dir_W
            suffix = "W"
        else:
            raise ValueError("??!?")

        # Maybe useful to know max hit dimension size from zeroth element
        clusters.sort(key=lambda cluster: -len(cluster.hits))

        event_data = []
        for cluster in clusters:
            cluster_data = make_cluster_data(cluster, event, view, args.hit_feature_preset)
            event_data.append(torch.tensor(cluster_data, dtype=torch.float32))
        # print(*(el.shape for el in event_data), sep="\n")
        # print(*(el for el in event_data), sep="\n")

        sim_mat = get_similarity_matrix(clusters)

        if args.save_mc_cnts:
            event_data = { "clusters" : event_data, "similarity" : sim_mat }
            event_data["cluster_mc_ids"] = [
                torch.tensor([ mc_id for mc_id in cluster.mc_id_cnt.keys() ], dtype=torch.long)
                for cluster in clusters
            ]
            event_data["cluster_mc_cnts"] = [
                torch.tensor([ mc_cnt for mc_cnt in cluster.mc_id_cnt.values() ], dtype=torch.long)
                for cluster in clusters
            ]
        else:
            event_data.append(sim_mat)

        torch.save(event_data, os.path.join(out_dir, "all", f"{i_event}_{suffix}.pt"))

def read_events(tree, n_events=None, skip_events=None):
    cluster_id = tree["cluster_id"].array(library="np")
    cluster_view = tree["cluster_view"].array(library="np")
    mc_id = tree["mc_id"].array(library="np")
    mc_pdg = tree["mc_pdg"].array(library="np")
    hit_cluster_id = tree["hit_cluster_id"].array(library="np")
    hit_mc_id = tree["hit_mc_id"].array(library="np")
    hit_x_rel_pos = tree["hit_x_rel_pos"].array(library="np")
    hit_z_rel_pos = tree["hit_z_rel_pos"].array(library="np")
    hit_r_rel_pos = tree["hit_r_rel_pos"].array(library="np")
    hit_ctheta_rel_pos = tree["hit_ctheta_rel_pos"].array(library="np")
    hit_stheta_rel_pos = tree["hit_stheta_rel_pos"].array(library="np")
    hit_x_width = tree["hit_x_width"].array(library="np")
    hit_x_gap_dist = tree["hit_x_gap_dist"].array(library="np")
    hit_energy = tree["hit_energy"].array(library="np")

    events = []
    skip_events = 0 if skip_events is None else skip_events
    max_events = tree.num_entries if n_events is None else skip_events + n_events
    for i in tqdm(range(skip_events, tree.num_entries), desc="Reading from ROOT file"):
        # -- XXX For ad-hoc _connected_accessory_clustering visualisation
        # if i != 49: continue; 
        # elif i > 49: break
        # -- XXX
        if i >= max_events:
            break
        
        events.append(
            Event(
                cluster_id[i],
                cluster_view[i],
                mc_id[i],
                mc_pdg[i],
                hit_cluster_id[i],
                hit_mc_id[i],
                hit_x_rel_pos[i],
                hit_z_rel_pos[i],
                hit_r_rel_pos[i],
                hit_ctheta_rel_pos[i],
                hit_stheta_rel_pos[i],
                hit_x_width[i],
                hit_x_gap_dist[i],
                hit_energy[i]
            )
        )

    return events

def main(args):
    events = read_events(uproot.open(args.filename)[args.treename], n_events=args.n_events)

    with tqdm_joblib(tqdm(total=len(events), desc="Building events + writing to disk")):
        joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(process_event)(i_ev, ev, args)
            for i_ev, ev in enumerate(events)
        )

def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("filename", type=str)
    parser.add_argument("treename", type=str)

    parser.add_argument("--cheated_feature", action="store_true")
    parser.add_argument(
        "--hit_feature_preset", type=int, default=1, choices=range(1, 9),
        help=(
            "1 - Cartesian | "
            "2 - Cartesian w/ cheat | "
            "3 - Cartesian w/ summary | "
            "4 - Polar | "
            "5 - Polar w/ summary | "
            "6 - Cartesian w/ wire pitch | "
            "7 - Cartesian + Polar w/ wire pitch | "
            "8 - Cartesian + Polar w/ wire pitch + View one-hot"
        )
    )
    parser.add_argument("--save_mc_cnts", action="store_true")

    parser.add_argument("--n_jobs", type=int, default=4)

    parser.add_argument("--n_events", type=int, default=None)

    parser.add_argument("--out_dir_U", type=str, default=None)
    parser.add_argument("--out_dir_V", type=str, default=None)
    parser.add_argument("--out_dir_W", type=str, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_cli())
