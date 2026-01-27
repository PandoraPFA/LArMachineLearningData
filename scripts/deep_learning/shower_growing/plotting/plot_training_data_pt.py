import argparse, os, itertools, random

import numpy as np
import matplotlib; from matplotlib import pyplot as plt
import torch

import sys; sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers import get_pitch

def main(args):
    for i_fname, fname in enumerate(os.listdir(args.data_dir)):
        if i_fname < args.skip_evs:
            continue
        if args.max_evs is not None and i_fname - args.skip_evs >= args.max_evs:
            break


        t_data = torch.load(os.path.join(args.data_dir, fname))
        t_in, t_target = t_data[:-1], t_data[-1]
        plot_name = fname.split(".pt")[0]
        print(f"{plot_name}:")
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        random.seed(0)
        def random_rgb():
            return (random.random(), random.random(), random.random())
        cs = { i_t_cluster : random_rgb() for i_t_cluster in range(len(t_in)) }

        first_colors = []
        for i_t_cluster, t_cluster in enumerate(t_in):
            c = cs[i_t_cluster]
            if i_t_cluster < 5:
                first_colors.append(c)
            for t_hit in t_cluster:
                if args.polar_coords:
                    r = t_hit[0].item()
                    c_theta = t_hit[1].item()
                    s_theta = t_hit[2].item()
                    x = c_theta * r
                    z = s_theta * r
                else:
                    x = t_hit[0].item()
                    z = t_hit[2].item()
                x_width = t_hit[args.x_width_idx].item()
                z_width = get_pitch(args.view)
                patch_corner = (x - (x_width / 2), z - (z_width / 2))
                ax[0].add_patch(
                    matplotlib.patches.Rectangle(
                        patch_corner, x_width, z_width, fill=False, edgecolor=c, linewidth=0.2
                    )
                )
        ax[0].scatter(0, 0, color="r", marker="x", s=6)
        ax[0].autoscale_view()
        # ax[0].set_ylim(-25, 300)
        # ax[0].set_xlim(-60, 60)
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("z")
        ax[0].set_title(f"Reco Clusters - {plot_name}")
        handles = [
            matplotlib.lines.Line2D(
                [], [], color="r", marker="x", linestyle="None", markersize=6, label="Reco Vertex"
            )
        ]
        handles += [
            matplotlib.patches.Patch(
                facecolor=first_colors[i], label=f"Cluster {i} - {len(t_in[i])} hits"
            )
            for i in range(len(first_colors))
        ]
        ax[0].legend(
            handles=handles, loc="upper center", title="5 Largest Clusters", ncols=2, fontsize=8
        )

        arr_target = t_target.numpy()
        # matrix_log = np.log10(arr_target + 1e06)
        cax = ax[1].imshow(arr_target, vmin=0, vmax=1, cmap="Greys")
        cbar = fig.colorbar(cax, ax=ax[1], fraction=0.046, pad=0.04)
        cbar.set_label("Similarity (0-1)")
        ax[1].set_title(f"Similarity Matrix, clusters ordered by num. hits ")
    
        plt.savefig(os.path.join(args.plot_save_dir, f"clusters-similarity_{plot_name}.pdf"))
        plt.close()

def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", type=str)
    parser.add_argument("plot_save_dir", type=str)

    parser.add_argument("--max_evs", default=None, type=int)
    parser.add_argument("--skip_evs", default=0, type=int)
    parser.add_argument("--view", default=6, type=int)
    parser.add_argument("--polar_coords", action="store_true")
    parser.add_argument("--x_width_idx", default=5)

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_cli())
