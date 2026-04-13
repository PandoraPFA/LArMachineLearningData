"""
Read in training data - reco clusters written out to a TTree from a pandora alg
"""

import argparse, os, random

import logging; logger = logging.getLogger("the_logger")

import numpy as np
from tqdm import tqdm; from tqdm_joblib import tqdm_joblib
import torch
import uproot
import joblib

import sys; sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers import get_pitch, setup_logging
from constants import SCALING_FACTORS, PITCHES, DETECTORS
from data.event import Event

def get_similarity_matrix(clusters, missing_mc_ids_in_denom=False):
    similarity_mat = torch.zeros(len(clusters), len(clusters), dtype=torch.float32)
    for i_cluster_a, cluster_a in enumerate(clusters):
        for i_cluster_b, cluster_b in enumerate(clusters):
            similarity = 0
            for mc_id, mc_cnt_a in cluster_a.mc_id_cnt.items(): # Dot product of purities wrt to MC particle "dimension"
                if mc_id < 0 or mc_id not in cluster_b.mc_id_cnt:
                    continue
                similarity += (
                    (
                        mc_cnt_a /
                        max(cluster_a.get_n_hits(with_mc_id=(not missing_mc_ids_in_denom)), 0)
                    ) *
                    (
                        cluster_b.mc_id_cnt[mc_id] /
                        max(cluster_b.get_n_hits(with_mc_id=(not missing_mc_ids_in_denom)), 0)
                    )
                )
            similarity_mat[i_cluster_a][i_cluster_b] = similarity

    return similarity_mat

def make_cluster_data(cluster, event, view, preset, detector):
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
                hit.x, hit.z, hit.x_width, get_pitch(view, detector), hit.x_gap_dist, hit.energy
            ]
            cluster_data.append(hit_feat_vec)

    elif preset == 7: # Cartesian + polar w/ wire pitch
        for hit in cluster.hits:
            hit_feat_vec = [
                hit.r, hit.c_theta, hit.s_theta,
                hit.x, hit.z, hit.x_width,
                get_pitch(view, detector), hit.x_gap_dist, hit.energy
            ]
            cluster_data.append(hit_feat_vec)

    elif preset == 8: # Cartesian + polar w/ wire pitch + view one-hot
        for hit in cluster.hits:
            hit_feat_vec = [
                hit.r, hit.c_theta, hit.s_theta,
                hit.x, hit.z, hit.x_width,
                get_pitch(view, detector), hit.x_gap_dist, hit.energy
            ]
            if view == 4:
                hit_feat_vec += [1., 0., 0.]
            elif view == 5:
                hit_feat_vec += [0., 1., 0.]
            else:
                hit_feat_vec += [0., 0., 1.]
            cluster_data.append(hit_feat_vec)

    elif preset == 9: # Cartesian + polar w/ wire pitch + view one-hot + pDUNE APA1 one-hot
        for hit in cluster.hits:
            assert hit.has_vol_id()

            hit_feat_vec = [
                hit.r, hit.c_theta, hit.s_theta,
                hit.x, hit.z, hit.x_width,
                get_pitch(view, detector), hit.x_gap_dist, hit.energy
            ]
            if view == 4:
                hit_feat_vec += [1., 0., 0.]
            elif view == 5:
                hit_feat_vec += [0., 1., 0.]
            else:
                hit_feat_vec += [0., 0., 1.]
            if hit.vol_id == (0, 0): # behind APA1
                hit_feat_vec += [1., 0.]
            elif hit.vol_id == (1, 0): # in front of APA1
                hit_feat_vec += [0., 1.]
            else:
                hit_feat_vec += [0., 0.]

            cluster_data.append(hit_feat_vec)

    elif preset == 10: # Cartesian + polar w/ wire pitch + view one-hot + no x_gap_dist
        for hit in cluster.hits:
            hit_feat_vec = [
                hit.r, hit.c_theta, hit.s_theta,
                hit.x, hit.z, hit.x_width,
                get_pitch(view, detector), hit.energy
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
            cluster_data = make_cluster_data(
                cluster, event, view, args.hit_feature_preset, args.detector
            )
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

def read_events(tree, n_events=None, shift_missing_mc_ids=False):
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
    if "hit_tpc_vol_id" in tree:
        assert "hit_daughter_vol_id" in tree
        hit_tpc_vol_id = tree["hit_tpc_vol_id"].array(library="np")
        hit_daughter_vol_id = tree["hit_daughter_vol_id"].array(library="np")
    else:
        hit_tpc_vol_id = [
            [ None for _ in range(len(hit_cluster_id[i])) ] for i in range(tree.num_entries)
        ]
        hit_daughter_vol_id = [
            [ None for _ in range(len(hit_cluster_id[i])) ] for i in range(tree.num_entries)
        ]
    if "mc_is_from_beam" in tree:
        mc_is_from_beam = tree["mc_is_from_beam"].array(library="np")
    else:
        mc_is_from_beam = [ [ 0 for _ in range(len(mc_id[i])) ] for i in range(tree.num_entries) ]

    events = []
    max_events = tree.num_entries if n_events is None else n_events
    randomiser = list(range(tree.num_entries))
    random.shuffle(randomiser)
    for _ in tqdm(range(max_events), desc="Reading from ROOT file"):
        i = randomiser.pop()
    
        if shift_missing_mc_ids and -1 in mc_id[i]:
            mc_id[i][mc_id[i] == -1] = -2
            hit_mc_id[i][hit_mc_id[i] == -1] = -2
        
        events.append(
            Event(
                cluster_id[i],
                cluster_view[i],
                mc_id[i],
                mc_pdg[i],
                mc_is_from_beam[i],
                hit_cluster_id[i],
                hit_mc_id[i],
                hit_x_rel_pos[i],
                hit_z_rel_pos[i],
                hit_r_rel_pos[i],
                hit_ctheta_rel_pos[i],
                hit_stheta_rel_pos[i],
                hit_x_width[i],
                hit_x_gap_dist[i],
                hit_energy[i],
                hit_tpc_vol_id[i],
                hit_daughter_vol_id[i]
            )
        )

    return events

def check_scaling_factors(filename, treename, detector):
    with uproot.open(filename) as f:
        event_treename = treename + "_event_data"
        if event_treename not in f:
            logger.warning(
                f"{event_treename} tree doesn't exist, not checking for consistent scaling factors"
            )
            return

        event_tree = f[event_treename]
        for name, scaling_factor in SCALING_FACTORS.items():
            if name.split(":")[0] != detector:
                continue
            for tree_scaling_factor in (
                event_tree["scalefactor_" + name.split(":")[1]].array(library="np")
            ):
                if f"{scaling_factor:.7g}" != f"{tree_scaling_factor:.7g}": # Is this bad?
                    raise ValueError(
                        f"Scaling factor {name} does not match factor expected by Pandora alg: "
                        f"{scaling_factor:.7g} vs {tree_scaling_factor:.7g}"
                    )
        logger.info("Scaling factors are consistent")

def check_pitches(filename, treename, detector):
    with uproot.open(filename) as f:
        view_treename = treename + "_view_data"
        if view_treename not in f:
            logger.warning(
                f"{view_treename} tree doesn't exist, not checking for consistent scaling factors"
            )
            return

        view_tree = f[view_treename]
        for view, pitch in zip(
            view_tree["view"].array(library="np"), view_tree["pitch"].array(library="np")
        ):
            if f"{pitch:.7g}" != f"{PITCHES[detector][view]:.7g}":
                raise ValueError(
                    f"Pitch for view {view} does not match pitch expected by Pandora alg: "
                    f"{PITCHES[detector][view]:.7g} vs {pitch:.7g}"
                )
        logger.info("Pitches are consistent")

def balance_events(events):
    beam_events = [ ev for ev in events if ev.beam_purity > 0.5 ]
    other_events = [ ev for ev in events if ev.beam_purity <= 0.5 ]
    n_keep = min(len(beam_events), len(other_events))
    random.shuffle(beam_events) # why not shuffle?
    random.shuffle(other_events)
    events = beam_events[:n_keep] + other_events[:n_keep]
    random.shuffle(events) # why not suffle some more?
    if len(beam_events) == 0:
        logger.warning("Found 0 beam events, 'mc_is_from_beam' may be missing from training tree")
    logger.info(
        f"Found {len(beam_events)} beam and {len(other_events)} non-beam events, "
        f"balancing events leaves {len(events)} events (50/50 beam non-beam split)"
    )
    return events

def main(args):
    if args.detector is not None:
        check_scaling_factors(args.filename, args.treename, args.detector)
        check_pitches(args.filename, args.treename, args.detector)
    else:
        logger.warning("'--detector' not specified, not checking for consistent scaling factors")

    events = read_events(
        uproot.open(args.filename)[args.treename],
        n_events=args.n_events, shift_missing_mc_ids=args.missing_mc_ids_in_denom,
    )

    if args.balance_events:
        events = balance_events(events)

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
        "--hit_feature_preset", type=int, default=1, choices=range(1, 11),
        help=(
            "1 - Cartesian | "
            "2 - Cartesian w/ cheat | "
            "3 - Cartesian w/ summary | "
            "4 - Polar | "
            "5 - Polar w/ summary | "
            "6 - Cartesian w/ wire pitch | "
            "7 - Cartesian + Polar w/ wire pitch | "
            "8 - Preset 7 + View one-hot | "
            "9 - Preset 8 + APA1 one-hot | "
            "10 - Preset 8 w/o X gap distance"
        )
    )
    parser.add_argument("--save_mc_cnts", action="store_true")
    parser.add_argument("--missing_mc_ids_in_denom", action="store_true")
    parser.add_argument("--balance_events", action="store_true")

    parser.add_argument("--n_jobs", type=int, default=4)

    parser.add_argument("--n_events", type=int, default=None)

    parser.add_argument("--detector", type=str, default=None, choices=DETECTORS)

    parser.add_argument("--out_dir_U", type=str, default=None)
    parser.add_argument("--out_dir_V", type=str, default=None)
    parser.add_argument("--out_dir_W", type=str, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    setup_logging()
    main(parse_cli())
