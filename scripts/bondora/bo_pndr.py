import subprocess, glob, os, time, argparse, sys, signal, psutil
import xml.etree.ElementTree as ET

import logging
logger = logging.getLogger("the_logger")

import numpy as np
import optuna

from config_parser import get_config

def suggest_and_set_params(trial, search_space, default_xml_path):
    config_tree = ET.parse(default_xml_path)
    root = config_tree.getroot()

    pars = {}
    for par_loc, par_data in search_space.items():
        par_type, par_min, par_max, par_step = par_data
        if par_type == "int":
            par_step = 1 if par_step is None else par_step
            par_val = trial.suggest_int(par_loc, par_min, par_max, step=par_step)
        elif par_type == "float" or par_type == "float_cosangle":
            par_val = trial.suggest_float(par_loc, par_min, par_max, step=par_step)
        elif par_type == "categorical_bool":
            par_val = trial.suggest_categorical(par_loc, par_step)
        else:
            raise NotImplementedError(f"Paramter type {par_type} not implemented")
        pars[par_loc] = par_val

        subalg_name = None
        if par_loc.count("/") == 1:
            alg_name, par_name = par_loc.split("/")
        elif par_loc.count("/") == 2:
            alg_name, subalg_name, par_name = par_loc.split("/")
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
                if par_type == "float_cosangle":
                    el.text = str(np.cos(np.deg2rad(par_val)))
                elif par_type == "categorical_bool":
                    el.text = "true" if par_val else "false"
                else:
                    el.text = str(par_val)

    config_tree.write(default_xml_path.split(".xml")[0] + "_TRIAL.xml")
    
    return pars

def objective(trial, conf):
    trial_params = suggest_and_set_params(trial, conf.search_space, conf.default_algs_xml_path)
    logger.info(
        f"Trial {trial.number} "
        f"{'(startup) ' if trial.number < conf.n_startup_trials else ''}start: {trial_params}"
    )

    for old_file in (
        glob.glob(os.path.join(conf.scratch_dir_path, "*.root")) +
        glob.glob(os.path.join(conf.scratch_dir_path, "*.txt"))
    ):
        os.remove(old_file)

    # Each processes does one file and picks up a new one once finished or timed-out and killed.
    # Continues until all files have been processes.
    wait_time = conf.expected_single_file_process_time * conf.n_files_per_process * 5
    procs = [None for _ in range(conf.n_processes)]
    proc_timeouts = [None for _ in range(conf.n_processes)]
    n_files_started, n_procs_started, n_files_overtime = 0, 0, 0
    while n_files_started < conf.n_files_total or any(p is not None for p in procs):
        now = time.time()
        for i_proc in range(conf.n_processes):
            # Under the limit of concurrent processes, start another one if files are left
            if procs[i_proc] is None:
                if n_files_started < conf.n_files_total:
                    procs[i_proc] = subprocess.Popen(
                        [
                            conf.pandora_run_script_path,
                                str(n_procs_started + 1),
                                str(conf.n_files_per_process),
                                conf.run_xml_path,
                                conf.pandora_setup_path,
                                conf.scratch_dir_path,
                                conf.pndr_data_path,
                                conf.pandora_geometry_path,
                                conf.pandora_interface_path
                        ],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        env=dict(os.environ, OMP_NUM_THREADS="1") # tell libtorch to chill
                    )
                    proc_timeouts[i_proc] = now + wait_time
                    n_files_started += conf.n_files_per_process
                    n_procs_started += 1
                continue

            # The process has finished
            if procs[i_proc].poll() is not None:
                procs[i_proc] = None
                proc_timeouts[i_proc] = None
                continue

            # The process is taking to long, will be killed
            if proc_timeouts[i_proc] < now:
                logger.info(f"{procs[i_proc].pid} pandora process taking too long, terminating :(")#)
                if not kill_pandora_proc(procs[i_proc]):
                    logger.info(f"Can't find {procs[i_proc].pid}, maybe it finished just in time?")
                    continue
                n_files_overtime += conf.n_files_per_process
                procs[i_proc] = None
                proc_timeouts[i_proc] = None

            time.sleep(0.1)

    # Config may result in clustering taking too long, scale down the mean rand index if this happens
    overtime_factor = 1 - (n_files_overtime / conf.n_files_total)

    subprocess.run(
        [conf.aggregate_validations_script_path, conf.pandora_setup_path, conf.scratch_dir_path],
        stdout=subprocess.DEVNULL
    )
    result_path = os.path.join(conf.scratch_dir_path, "result.txt")
    
    conf.result_parser.on_new_trial(trial, result_path)
    result = conf.result_parser.parse(result_path, overtime_factor)

    return result

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

def kill_pandora_proc(proc): # Pandora made me kill the children /0-0\
    try:
        ps_proc = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        logger.info(f"Can't find {proc.pid}, maybe it finished just in time?")
        return False
    for child_proc in ps_proc.children(recursive=True):
        child_proc.send_signal(signal.SIGTERM)
    ps_proc.send_signal(signal.SIGTERM)
    return True

def main(args):
    setup_logging()
    conf = get_config(args.config_file)

    if conf.sampler == "gp":
        sampler = optuna.samplers.GPSampler(
            n_startup_trials=conf.n_startup_trials, seed=conf.sampler_seed
        )
    elif conf.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=conf.sampler_seed)
    else:
        raise ValueError(f"sampler '{conf.sampler}' not valid")

    study = optuna.create_study(
        study_name=conf.study_name,
        storage=conf.study_storage_db_path,
        sampler=sampler,
        load_if_exists=conf.load_study,
        direction="maximize"
    )
    if conf.initial_param_point is not None:
        study.enqueue_trial(conf.initial_param_point, user_attrs={ "is_default" : True })
        logger.info("First trial will be configured at initial parameters")
    while True:
        trial = study.ask()
        result = objective(trial, conf)
        study.tell(trial, result)
        logger.info(
            f"Trial {trial.number} complete with score {result}. "
            f"Best is trial {study.best_trial.number} with score {study.best_value}."
        )
        if trial.number >= conf.n_trials:
            break

    logger.info(
        f"Best is trial {study.best_trial.number} "
        f"with score {study.best_value} for params {study.best_params}."
    )

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
