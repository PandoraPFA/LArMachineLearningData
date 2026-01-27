import argparse, itertools
from collections import defaultdict

import numpy as np
import matplotlib; from matplotlib import pyplot as plt
import uproot

class Event:
    def __init__(
        self,
        cluster_ids, cluster_views,
        mc_ids, mc_pdgs,
        hit_cluster_ids, hit_xs, hit_x_widths, hit_zs, hit_z_widths, hit_energies, hit_mc_ids
    ):
        self.mcs = { id : pdg for id, pdg in zip(mc_ids, mc_pdgs) }

        hits = defaultdict(list)
        for x, x_width, z, z_width, energy, mc_id, cluster_id in zip(
            hit_xs, hit_x_widths, hit_zs, hit_z_widths, hit_energies, hit_mc_ids, hit_cluster_ids
        ):
            hit = Hit(x, x_width, z, z_width, energy)
            hit.add_main_mc(mc_id, self.mcs[mc_id])
            hits[cluster_id].append(hit)
            
        self.view_clusters = defaultdict(list)
        for id, view in zip(cluster_ids, cluster_views):
            cluster = Cluster(id, view)
            for hit in hits[id]:
                cluster.add_hit(hit)
            self.view_clusters[view].append(cluster)

        # print("New Event:")
        # mc_pdgs = { id : self.mcs[id] for id in sorted(self.mcs.keys()) }
        # print(f"mcs: {mc_pdgs}")
        # for view, clusters in view_clusters.items():
        #     print(f"  {view}:")
        #     print(f"    {len(clusters)} clusters:")
        #     mcs = set()
        #     for cluster in clusters:
        #         mc_dict = defaultdict(int)
        #         for hit in cluster.hits:
        #             mc_dict[hit.main_mc_id] += 1
        #         print(f"      {len(cluster.hits)} hits ({dict(mc_dict)})")
        
class Cluster:
    def __init__(self, id, view):
        self.id = id
        self.view = view
        self.hits = []
        self.main_mc_id = None

    def add_hit(self, hit):
        self.hits.append(hit)
        self.main_mc_id = None

    def calc_main_mc(self):
        mc_id_cnt = defaultdict(int)
        for hit in self.hits:
            mc_id_cnt[hit.main_mc_id] += 1
        self.main_mc_id = max(mc_id_cnt.keys(), key=lambda k: mc_id_cnt[k])

class Hit:
    def __init__(self, x, x_width, z, z_width, energy):
        self.x = x
        self.x_width = x_width
        self.z = z
        self.z_width = z_width
        self.energy = energy
        self.main_mc_id = None
        self.main_mc_pdg = None
        # self.patch_corner = (z - (z_width / 2), x - (x_width / 2))
        self.patch_corner = (x - (x_width / 2), z - (z_width / 2))

    def add_main_mc(self, id, pdg):
        self.main_mc_id = id
        self.main_mc_pdg = pdg

def read_events(tree, max_evs, skip_evs):
    events = []
    for i in range(tree.num_entries):
        if skip_evs is not None and i < skip_evs:
            continue
        events.append(
            Event(
                np.array(tree["cluster_id"].array()[i]),
                np.array(tree["cluster_view"].array()[i]),
                np.array(tree["mc_id"].array()[i]),
                np.array(tree["mc_pdg"].array()[i]),
                np.array(tree["hit_cluster_id"].array()[i]),
                np.array(tree["hit_x_rel_pos"].array()[i]),
                np.array(tree["hit_x_width"].array()[i]),
                np.array(tree["hit_z_rel_pos"].array()[i]),
                np.array(tree["hit_z_width"].array()[i]),
                np.array(tree["hit_energy"].array()[i]),
                np.array(tree["hit_mc_id"].array()[i])
            )
        )
        if len(events) >= max_evs:
            break
    return events

def main(args):
    events = read_events(uproot.open(args.filename)[args.treename], args.max_evs, args.skip_evs)

    # TODO 
    # - plot hits as rectangles with centre got for true shower MC particle, centre square for true track MC particle
    # - Colour of hit rectangle as the reco cluster it is in
    # - Colour of hit rectangle as the main MC particle of the cluster (ignore draws for now)

    for i_event, event in enumerate(events):
        print(f"Event {i_event}")
        for view, clusters in event.view_clusters.items():
            print(f"  View {view}")

            fig, ax = plt.subplots(1, 2, figsize=(12, 8))
            c_iter = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            for cluster in clusters:
                c = next(c_iter)
                for hit in cluster.hits:
                    ax[0].add_patch(
                        matplotlib.patches.Rectangle(
                            hit.patch_corner, hit.x_width, hit.z_width, fill=False, edgecolor=c
                        )
                    )
                    if abs(hit.main_mc_pdg) == 11 or abs(hit.main_mc_pdg) == 22:
                        ax[0].add_patch(
                            matplotlib.patches.RegularPolygon(
                                (hit.x, hit.z),
                                numVertices=3,
                                radius=hit.z_width / 6,
                                transform=ax[0].transData,
                                color=c
                            )
                        )
                    else:
                        ax[0].add_patch(
                            matplotlib.patches.Rectangle(
                                (hit.x - (hit.z_width / 6), hit.z - (hit.z_width / 24)),
                                hit.z_width / 3,
                                hit.z_width / 12,
                                color=c
                            )
                        )
            ax[0].autoscale_view()
            ax[0].set_xlabel("x")
            ax[0].set_ylabel("z")
            ax[0].set_title(f"Reco Clusters - View {view}")

            mc_hits = defaultdict(list)
            c_iter = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            for cluster in clusters:
                for hit in cluster.hits:
                    mc_hits[hit.main_mc_id].append(hit)
            for hits in mc_hits.values():
                c = next(c_iter)
                for hit in hits:
                    ax[1].add_patch(
                        matplotlib.patches.Rectangle(
                            hit.patch_corner, hit.x_width, hit.z_width, fill=False, edgecolor=c
                        )
                    )
                    if abs(hit.main_mc_pdg) == 11 or abs(hit.main_mc_pdg) == 22:
                        ax[1].add_patch(
                            matplotlib.patches.RegularPolygon(
                                (hit.x, hit.z),
                                numVertices=3,
                                radius=hit.z_width / 6,
                                transform=ax[1].transData,
                                color=c
                            )
                        )
                    else:
                        ax[1].add_patch(
                            matplotlib.patches.Rectangle(
                                (hit.x - (hit.z_width / 6), hit.z - (hit.z_width / 24)),
                                hit.z_width / 3,
                                hit.z_width / 12,
                                color=c
                            )
                        )
            ax[1].autoscale_view()
            ax[1].set_xlabel("x")
            ax[1].set_ylabel("z")
            ax[1].set_title(f"True Clusters - View {view}")

            print(f"    Reco Clusters: {len(clusters):<5} True Clusters: {len(mc_hits):<5}")

            plt.show()


def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("filename", type=str)
    parser.add_argument("treename", type=str)

    parser.add_argument("--max_evs", default=None, type=int)
    parser.add_argument("--skip_evs", default=None, type=int)

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_cli())
