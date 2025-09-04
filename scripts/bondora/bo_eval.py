import argparse, os
import xml.etree.ElementTree as ET

from matplotlib import pyplot as plt
import matplotlib
import numpy as np

import logging
logger = logging.getLogger("the_logger")

import optuna
from optuna.visualization.matplotlib import (
    plot_param_importances,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_rank,
    plot_slice,
    plot_contour,
    plot_timeline
)

from bo_pndr import setup_logging
from config_parser import get_config

SAVE_DIR="/storage/epp2/phsajw/optuna_bo/plots" # ATTN change this!

def main(args):
    setup_logging()
    conf = get_config(args.config_file)

    sampler = optuna.samplers.GPSampler(n_startup_trials=conf.n_startup_trials)

    study = optuna.create_study(
        study_name=conf.study_name,
        storage=conf.study_storage_db_path,
        sampler=sampler,
        load_if_exists=True,
        direction="maximize"
    )
    logger.info(f"Loaded study from {conf.study_storage_db_path}")
    logger.info(f"Total trials is {len(study.trials)}")

    default_trial = None
    if args.has_default_trial:
        for trial in study.trials:
            if trial.number == 0:
                default_trial = trial
                break
        logger.info(
            f"Default configuration has score {trial.value} at params:\n" +
            get_params_str(default_trial.params, conf)
        )

    best_trial = None
    if args.get_best_trial or args.get_best_trial_xml or args.plot_param_importance: # Want best_trial for param importance plot
        best_trial = study.best_trial
        out_str = (
            f"Best trial is {study.best_trial.number} "
            f"with score {get_score_str(study.best_value, default_trial)} at params:\n" +
            get_params_str(study.best_params, conf)
        )
        if args.has_default_trial:
            out_str += (
                "\n    Param distance to default configuration is "
                f"{get_param_distance_to_default(study.best_params, default_trial)}"
            )
        logger.info(out_str)

        equivalent_trials = []
        for trial in study.trials:
            if trial.value is None or trial.value < study.best_value or trial.number == study.best_trial.number:
                continue
            equivalent_trials.append(trial)
        if len(equivalent_trials) > 0:
            logger.info(f"{len(equivalent_trials)} later trials have the same score")
            if args.has_default_trial:
                best_distance =  get_param_distance_to_default(best_trial.params, default_trial)
                for trial in equivalent_trials:
                    distance = get_param_distance_to_default(trial.params, default_trial)
                    if distance < best_distance:
                        best_trial, best_distance = trial, distance
                if best_trial.number != study.best_trial.number:
                    out_str = (
                        "Another best trial closer to the default parameters is " +
                        f"trial {best_trial.number} at params:\n" +
                        get_params_str(best_trial.params, conf) +
                        f"\n    Param distance to default configuration is {best_distance}"
                    )
                    logger.info(out_str)
                else:
                    logger.info("No later trials are closer to the default configuration")

        if args.get_best_trial_xml:
            make_xml(best_trial.params, conf)

    if args.get_best_n_trials is not None:
        out_str = f"Best {args.get_best_n_trials} trials are:\n"
        # Crashes midway through an optimisation creates a null value
        trials = [ trial for trial in study.trials if trial.value is not None ]
        trials.sort(key=lambda trial: trial.value, reverse=True)
        for trial in trials[:args.get_best_n_trials]:
            out_str += (
                f"Trial {trial.number} " +
                f"with score {get_score_str(trial.value, default_trial)} at params\n" +
                get_params_str(trial.params, conf)
            )
            if args.has_default_trial:
                out_str += (
                    "\n    Param distance to default configuration is "
                    f"{get_param_distance_to_default(trial.params, default_trial)}"
                )
            out_str += "\n"
        logger.info(out_str)

    if args.plot_param_importance or args.get_param_importance:
        ax = plot_param_importances(study, params=args.plot_params)
        if args.get_param_importance:
            importances = reversed(ax.containers[0].datavalues)
            params = reversed([ text.get_text() for text in ax.get_yticklabels() ])
            out_str = (
                "Parameter importances are:\n" +
                "\n".join(f"    {r:<2}. {i:.4f} - {p}" for r, (p, i) in enumerate(zip(params, importances)))
            )
            logger.info(out_str)
        if args.plot_param_importance:
            plt.close()
            importances = list(reversed(ax.containers[0].datavalues))
            params = list(reversed([ text.get_text() for text in ax.get_yticklabels() ]))
            default_vals = [ default_trial.params[param] for param in params ]
            optimised_vals = [ best_trial.params[param] for param in params ]
            optimised_vals_rel = [
                optim_val / (default_val if default_val != 0 else 1)
                for optim_val, default_val in zip(optimised_vals, default_vals)
            ]
            default_vals_rel = [ 1 for _ in range(len(default_vals)) ]

            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            logger.warning("Reverting back to the default mpl stylesheet!")
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            x = np.arange(len(params))
            width = 0.25
            spacing = 0.0
            rects1 = ax.bar(
                x - (width + spacing), default_vals_rel, width,
                fill=False, facecolor="C0", edgecolor="C0", hatch="xx", label="Default"
            )
            rects2 = ax.bar(
                x, optimised_vals_rel, width,
                facecolor="C0", edgecolor="C0", label="Optimised"
            )
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            ax.set_ylabel("Param. Relative to Default", loc="top", weight="bold", fontsize=12)
            ax.yaxis.set_tick_params(colors="C0", which="both")
            ax.yaxis.label.set_color("C0")
            ax.set_ylim(top=max(optimised_vals_rel + [1]) * 1.2)
            ax.set_xticks(x, [ param.split("/")[-1] for param in params ], rotation=30, fontsize=10)

            ax2 = ax.twinx()
            rects3 = ax2.bar(
                x + (width + spacing), importances, width,
                facecolor="C1", edgecolor="C1", label="Importance"
            )
            ax2.set_ylabel("Param. Importance", loc="top", weight="bold", fontsize=12)
            ax2.yaxis.set_tick_params(colors="C1", which="both")
            ax2.yaxis.label.set_color("C1")
            ax2.set_ylim(top=max(importances) * 1.2)

            ax_ylims = ax.axes.get_ylim()
            ax_yratio = ax_ylims[0] / ax_ylims[1]
            ax2_ylims = ax2.axes.get_ylim()
            ax2_yratio = ax2_ylims[0] / ax2_ylims[1]
            if ax_yratio < ax2_yratio:
                ax2.set_ylim(bottom = ax2_ylims[1]*ax_yratio)
            else:
                ax.set_ylim(bottom = ax_ylims[1]*ax2_yratio)

            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles += handles2
            labels += labels2
            ax.legend(handles, labels, loc="upper center", ncols=3, fontsize=12)

            autolabel(rects1, default_vals, ax)
            autolabel(rects2, optimised_vals, ax)
            autolabel(rects3, importances, ax2, high_precision=True)

            fig.tight_layout()
            make_plot(plt, conf.study_name + "_optmised_params.pdf", args.save_plots)
        else:
            plt.close()

    if args.plot_history:
        ax = plot_optimization_history(study)
        max_xs = ax.lines[0].get_xdata()
        max_ys = ax.lines[0].get_ydata()
        trial_xs = [ xy[0] for xy in ax.collections[0].get_offsets() ]
        trial_ys = [ xy[1] for xy in ax.collections[0].get_offsets() ]
        if True: # Remove outliers (usually when overtime factor has been applied)
            obj_val_threshold = np.percentile(trial_ys, 0.5)
            trial_xs = [ x for x, y in zip(trial_xs, trial_ys) if y > obj_val_threshold ]
            trial_ys = [ y for y in trial_ys if y > obj_val_threshold ]
        plt.close()
        _, ax = plt.subplots(1, 1, figsize=(10,6))
        ax.plot(max_xs, max_ys, c="r", label="Maximum", alpha=0.4)
        ax.scatter(trial_xs[0], trial_ys[0], c="k", marker="x", label="Default Trial")
        ax.scatter(trial_xs[1:], trial_ys[1:], c="b", label="Trial", s=8, alpha=0.8)
        plt.legend(loc="lower right")
        ax.set_xlabel("Trial Number", loc="right")
        ax.set_ylabel("Objective Value", loc="top")
        make_plot(plt, conf.study_name + "_optimisation_history.pdf", args.save_plots)

    if args.plot_parallel_coord:
        ax = plot_parallel_coordinate(study, params=args.plot_params)
        plt.show()

    if args.plot_rank:
        ax = plot_rank(study, params=args.plot_params)
        plt.show()

    if args.plot_slice:
        ax = plot_slice(study, params=args.plot_params)
        plt.show()

    if args.plot_contour:
        if args.plot_params is None:
            ax = plot_contour(study, params=args.plot_params)
            plt.show()
        else:
            ax = plot_contour(study, params=args.plot_params)

            obj_val_threshold = np.percentile([ trial.value for trial in study.trials ], 0.5)
            logger.info(f"Using min. threshold of {obj_val_threshold} for objective value")

            N = len(args.plot_params)
            for i in range(N):
                for j in range(N):
                    a = ax[i, j]

                    col_param_name = a.get_xlabel()
                    
                    if i == 0 and j == 0:
                        a.set_ylabel("Objective Value", fontsize=10)
                    elif j == 0:
                        a.set_ylabel(a.get_ylabel().split("/")[-1], fontsize=10)
                    else:
                        a.set_ylabel(None)
                    if i == N - 1:
                        a.set_xlabel(a.get_xlabel().split("/")[-1], fontsize=10)
                    else:
                        a.set_xlabel(None)
                    
                    if j == i:
                        x, y = [], []
                        for trial in study.trials:
                            if trial.value < obj_val_threshold:
                                continue
                            x.append(trial.params[col_param_name])
                            y.append(trial.value)
                        a.scatter(x, y, c="k", s=4)
                        a.autoscale()
                        continue

                    for artist in a.collections: # Remove scatter points
                        if isinstance(artist, matplotlib.collections.PathCollection):
                            artist.remove()

            fig = ax[0, 0].figure
            fig.set_size_inches(N * 4, N * 3)
            make_plot(plt, conf.study_name + "_contour.pdf", args.save_plots)

    if args.plot_timeline:
        ax = plot_timeline(study)
        plt.show()

def get_score_str(score, default_trial):
    if default_trial is None:
        return str(score)
    return f"{str(score)} ({score - default_trial.value:+g})"

def get_params_str(params, conf):
    out_dict = {}
    for key, val in params.items():
        if conf.search_space[key][0] == "float_cosangle":
            out_dict[key] = f"{val} ({np.cos(np.deg2rad(val))})"
        else:
            out_dict[key] = f"{val}"
    return "\n".join(f"    - {key} = {val}" for key, val in out_dict.items())

def make_xml(params, conf):
    config_tree = ET.parse(conf.default_algs_xml_path)
    root = config_tree.getroot()
    for key, val in params.items():

        subalg_name = None
        if key.count("/") == 1:
            alg_name, par_name = key.split("/")
        elif key.count("/") == 2:
            alg_name, subalg_name, par_name = key.split("/")
        else:
            raise NotImplementedError("xml parameters nested deeper than 2 alg tags not supported")

        for node in root:
            if node.tag != "algorithm" or node.attrib["type"] != alg_name:
                continue

            if subalg_name is not None:
                for subnode in node:
                    if subnode.tag == "algorithm" and subnode.attrib["type"] == subalg_name:
                        node = subnode
                        break

            for el in node:
                if el.tag != par_name:
                    continue
                if conf.search_space[key][0] == "float_cosangle":
                    el.text = str(np.cos(np.deg2rad(val)))
                elif conf.search_space[key][0] == "categorical_bool":
                    el.text = "true" if val else "false"
                else:
                    el.text = str(val)

    config_tree.write(conf.default_algs_xml_path.split(".xml")[0] + "_BESTTRIAL.xml")
    logger.info(f"Written xml: {conf.default_algs_xml_path.split('.xml')[0] + '_BESTTRIAL.xml'}")

def get_param_distance_to_default(params, default_trial):
    distances = []
    for key, val in params.items():
        if type(val) is bool:
            delta = 0 if val == default_trial.params[key] else 1
            continue
        delta = val - default_trial.params[key]
        if delta == 0:
            distances.append(delta)
        elif delta > 0:
            distances.append(delta / (default_trial.distributions[key].high - default_trial.params[key]))
        else:
            distances.append(delta / (default_trial.params[key] - default_trial.distributions[key].low))
    return np.sqrt(sum(d**2 for d in distances)) / len(distances)

def make_plot(plt, savename, save_plot):
    if save_plot:
        plt.savefig(os.path.join(SAVE_DIR, savename))
        plt.close()
    else:
        plt.show()

def autolabel(rects, vals, ax, high_precision=False):
    for rect, val in zip(rects, vals):
        height = rect.get_height()
        ax.annotate(
            f"{val:.3f}" if high_precision else f"{val:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=90,
            weight="bold"
        )

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file")

    parser.add_argument("--has_default_trial", action="store_true")

    parser.add_argument("--get_best_trial", action="store_true")
    parser.add_argument("--get_best_trial_xml", action="store_true")
    parser.add_argument("--get_best_n_trials", type=int, default=None)

    parser.add_argument("--plot_param_importance", action="store_true")
    parser.add_argument("--get_param_importance", action="store_true")
    parser.add_argument("--plot_history", action="store_true")
    parser.add_argument("--plot_parallel_coord", action="store_true")
    parser.add_argument("--plot_rank", action="store_true")
    parser.add_argument("--plot_slice", action="store_true")
    parser.add_argument("--plot_timeline", action="store_true")
    parser.add_argument("--plot_contour", action="store_true")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("-p", "--plot_params", type=str, action="append", default=None)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
