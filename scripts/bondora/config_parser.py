import os, shutil
import xml.etree.ElementTree as ET
from collections import namedtuple

import logging
logger = logging.getLogger("the_logger")

import yaml

from result_parsers import result_parsers

defaults = {
    "initial_param_point" : None,
    "study_storage_name" : None,
    "load_study" : False,
    "sampler_seed": None,
    "sampler": "gp"
}

mandatory_fields = {
    "study_name",
    "search_space",
    "n_trials",
    "n_startup_trials",
    "n_processes",
    "n_files_per_process",
    "n_files_total",
    "expected_single_file_process_time",
    "result_parser",
    "scratch_dir_path",
    "studies_dir",
    "pndr_data_path",
    "bondora_base_path",
    "default_algs_xml_relpath",
    "run_xml_relpath",
    "pandora_run_script_relpath",
    "aggregate_validations_script_relpath",
    "pandora_base_path",
    "pandora_setup_relpath",
    "pandora_interface_relpath",
    "pandora_geometry_relpath"
}

def get_config(conf_path, _overwrite_dict={}):
    logger.info(f"Reading study configuration from {conf_path}")

    with open(conf_path, "r") as f:
        conf_dict = yaml.load(f, Loader=yaml.FullLoader)

    for field, val in _overwrite_dict.items(): # for hacking purposes
        conf_dict[field] = val

    missing_fields = mandatory_fields - set(conf_dict.keys())
    if missing_fields:
        raise ValueError(f"Missing mandatory fields {missing_fields} in conf file at {conf_path}")

    for option in set(defaults.keys()) - set(conf_dict.keys()):
        conf_dict[option] = defaults[option]

    _make_paths(conf_dict)

    # Prep results dir and study DB
    if conf_dict["study_storage_name"] is not None:
        study_dir = os.path.join(conf_dict["studies_dir"], conf_dict["study_name"])
        if not os.path.exists(study_dir):
            logger.info(f"Creating study directory {study_dir} to store study results")
            os.makedirs(study_dir)
        shutil.copyfile(conf_path, os.path.join(study_dir, os.path.basename(conf_path)))
        conf_dict["study_storage_db_path"] = os.path.join(
            "sqlite:///" + study_dir, conf_dict["study_storage_name"]
        )
    else:
        conf_dict["study_storage_db_path"] = None
    del conf_dict["study_storage_name"]

    # Prep scratch dir
    scratch_dir = os.path.join(conf_dict["scratch_dir_path"], conf_dict["study_name"])
    if not os.path.exists(scratch_dir):
        logger.info(f"Creating scartch directory {scratch_dir}")
        os.makedirs(scratch_dir)
    conf_dict["scratch_dir_path"] = scratch_dir

    # Prep Pandora xmls in the scratch dir
    scratch_run_xml_path = os.path.join(
        conf_dict["scratch_dir_path"], os.path.basename(conf_dict["run_xml_path"])
    )
    shutil.copyfile(conf_dict["run_xml_path"], scratch_run_xml_path)
    scratch_default_algs_xml_path = os.path.join(
        conf_dict["scratch_dir_path"], os.path.basename(conf_dict["default_algs_xml_path"])
    )
    shutil.copyfile(conf_dict["default_algs_xml_path"], scratch_default_algs_xml_path)
    run_tree = ET.parse(scratch_run_xml_path)
    run_root = run_tree.getroot()
    for node in run_root:
        if node.tag != "algorithm" or node.attrib["type"] != "LArDLMaster":
            continue
        for el in node:
            if el.tag != "NuSettingsFile":
                continue
            el.text = scratch_default_algs_xml_path.split(".xml")[0] + "_TRIAL.xml"
            break
        break
    run_tree.write(scratch_run_xml_path)
    conf_dict["run_xml_path"] = scratch_run_xml_path
    conf_dict["default_algs_xml_path"] = scratch_default_algs_xml_path

    if conf_dict["result_parser"] == "trackshower":
        if conf_dict["initial_param_point"] is None:
            raise ValueError("The 'trackshower' result parser requires `initial_param_point`")
        logger.warning(
            "Using 'trackshower' result parser, "
            "expecting 'initial_param_point' to be the default/baseline configuration parameters"
            )
    try:
        conf_dict["result_parser"] = result_parsers[conf_dict["result_parser"]]
    except KeyError:
        raise ValueError(
            f"result parser '{conf_dict['result_parser']}' invalid. "
            f"Valid parsers are {result_parsers}"
        )

    if conf_dict["n_files_total"] % conf_dict["n_files_per_process"] != 0:
        raise ValueError("'n_files_total' should be divisible by 'n_files_per_process'")

    conf_namedtuple = namedtuple("conf", conf_dict)
    conf = conf_namedtuple(**conf_dict)

    return conf

def _make_paths(conf_dict):
    def join(d, basepath, relpath, path):
        d[path] = os.path.join(d[basepath], d[relpath])

    join(conf_dict, "pandora_base_path", "pandora_setup_relpath", "pandora_setup_path")
    join(conf_dict, "pandora_base_path", "pandora_interface_relpath", "pandora_interface_path")
    join(conf_dict, "pandora_base_path", "pandora_geometry_relpath", "pandora_geometry_path")

    join(conf_dict, "bondora_base_path", "run_xml_relpath", "run_xml_path")
    join(conf_dict, "bondora_base_path", "default_algs_xml_relpath", "default_algs_xml_path")
    join(conf_dict, "bondora_base_path", "pandora_run_script_relpath", "pandora_run_script_path")
    join(conf_dict, "bondora_base_path", "aggregate_validations_script_relpath", "aggregate_validations_script_path")
