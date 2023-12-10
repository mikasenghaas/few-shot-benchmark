"""
Module containing helper functions to be used in run.py
"""
import glob
import os
import random
import logging
from typing import Dict, Tuple

import numpy as np
import hydra
from prettytable import PrettyTable
import torch
from omegaconf import OmegaConf, open_dict
from torch import nn

from typing import Union


def get_assigned_file(checkpoint_dir: str, num: int) -> str:
    """
    Get checkpoint file path from checkpoint directory and iteration number.

    Args:
        checkpoint_dir: str
        num: int

    Returns:
        assign_file: str
    """
    assign_file_path = os.path.join(checkpoint_dir, f"{num}.tar")
    return assign_file_path


def get_resume_file(checkpoint_dir: str) -> str:
    """
    Get latest checkpoint file path from checkpoint directory.

    Args:
        checkpoint_dir: str

    Returns:
        resume_file_path: str
    """
    filelist = glob.glob(os.path.join(checkpoint_dir, "*.tar"))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != "best_model.tar"]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file_path = os.path.join(checkpoint_dir, "{:d}.tar".format(max_epoch))
    return resume_file_path


def get_best_file(checkpoint_dir: str) -> str:
    """
    Get best checkpoint file path from checkpoint directory.

    Args:
        checkpoint_dir: str

    Returns:
        best_file_path: str
    """
    best_file_path = os.path.join(checkpoint_dir, "best_model.tar")
    if os.path.isfile(best_file_path):
        return best_file_path
    else:
        return get_resume_file(checkpoint_dir)


def get_latest_dir(checkpoint_dir: str) -> str:
    """
    Get latest checkpoint directory from checkpoint directory where each
    checkpoint directory has a name like yyyymmdd_hhmmss.

    Args:
        checkpoint_dir: str

    Returns:
        latest_dir: str
    """
    dirlist = glob.glob(os.path.join(checkpoint_dir, "*"))
    if len(dirlist) == 0:
        return ValueError("checkpoint dir not found")
    latest_dir = sorted(dirlist)[-1]

    return latest_dir


def get_model_file(cfg: OmegaConf) -> str:
    """
    Get checkpoint file path from config.

    Args:
        cfg: OmegaConf

    Returns:
        model_file: str
    """
    cp_cfg = cfg.checkpoint
    if cp_cfg.time == "latest":
        dir = get_latest_dir(cp_cfg.dir)
    else:
        dir = os.path.join(cp_cfg.dir, cp_cfg.time)

    # print(f"Using checkpoint dir: {dir}")
    return get_assigned_file(dir, cp_cfg.test_iter)


def fix_seed(seed=42) -> None:
    """
    Set random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    random.seed(seed)


def model_to_dict(model: nn.Module) -> dict:
    """
    Recursively convert a model to a dictionary.

    Args:
        model: nn.Module

    Returns:
        model_dict: dict
    """
    if isinstance(model, nn.Module):
        model_dict = {}
        children = list(model.named_children())
        if len(children) > 0:
            for name, module in children:
                model_dict[name] = model_to_dict(module)
        else:
            return str(model)
        return model_dict
    else:
        return str(model)


def opt_to_dict(opt: torch.optim.Optimizer) -> dict:
    """
    Convert optimiser to dictionary.

    Args:
        opt: torch.optim.Optimizer

    Returns:
        opt_dict: dict
    """
    opt_dict = opt.param_groups[0].copy()
    opt_dict.pop("params")
    return opt_dict


def get_exp_name(cfg: OmegaConf) -> Tuple[str, Dict]:
    """
    Returns the experiment name to be used for logging and checkpointing and
    all the parameters that make up the experiment name.
    """
    method = cfg.method.name
    dataset = cfg.dataset.name
    use_sot = cfg.use_sot
    n_way = cfg.n_way
    n_shot = cfg.n_shot
    exp_name = (
        (f"{method}-{dataset}{'-sot' if use_sot else ''}-{n_way}-way-{n_shot}-shot")
        if not cfg.name
        else cfg.name
    )
    exp_params = {
        "method": method,
        "dataset": dataset,
        "use_sot": use_sot,
        "n_way": n_way,
        "n_shot": n_shot,
    }

    return exp_name, exp_params


def hydra_setup(cfg) -> None:
    """
    Setup hydra and add hydra logging directory and experiment name to config.

    TODO: Figure exactly what this does.
    """
    os.environ["HYDRA_FULL_ERROR"] = "1"

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        # Add hydra logging directory to config
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        cfg.paths.log_dir = hydra_cfg["runtime"]["output_dir"]

        # Add experiment name to config
        cfg.name = get_exp_name(cfg)[0]

    try:
        OmegaConf.register_new_resolver("mul", lambda x, y: float(x) * float(y))
    except Exception as _:
        pass


def check_cfg(cfg: OmegaConf) -> None:
    """
    Check that the config is valid. Raises ValueError if not.

    Args:
        cfg: OmegaConf
    """
    mandatory_keys = [
        "group",
        "method",
        "dataset",
        "use_sot",
        "n_way",
        "n_shot",
    ]
    for key in mandatory_keys:
        assert key in cfg, f"Missing mandatory key: {key}"


def get_device(device: Union[str, None] = None) -> torch.device:
    """
    Get device to train on. If device is specified, use that. Otherwise, use
    the first available device from the following list: ["cuda", "mps", "cpu"].

    Returns:
        device: torch.device
    """
    if device:
        return torch.device(device)

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def filter_keys(d: dict) -> dict:
    """
    Recursively filter out all keys starting with "_" for
    nicer logging.

    Args:
        d: dict

    Returns:
        cfg: OmegaConf
    """
    if isinstance(d, dict):
        return {
            key: filter_keys(value)
            for key, value in d.items()
            if not key.startswith("_")
        }
    return d


def print_cfg(cfg: OmegaConf):
    """
    Print the experiment configuration using the PrettyTable library.

    Args:
        cfg: OmegaConf

    Returns:
        None
    """
    # Get experiment name and parameters
    exp_name, exp_params = get_exp_name(cfg)
    print(f"Running experiment: {exp_name}\n")

    # Print experiment parameters
    table = PrettyTable()
    table.field_names = exp_params.keys()
    table.add_row(exp_params.values())
    print(table)

    print(f"\n📣 For full configuration, see {cfg.paths.log_dir}/.hydra/config.yaml\n\n")


def get_logger(name: str, cfg: OmegaConf) -> logging.Logger:
    """
    Create a custom logger with specific formatting.

    Args:
        name: str, name of the logger
        level: logging level

    Returns:
        logger: logging.Logger
    """
    # Create or get the logger
    logger = logging.getLogger(name)

    # Set the level
    level = logging.getLevelName(cfg.general.log_level)
    logger.setLevel(level)

    # Configure the logger only once
    logger.handlers.clear()

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("[%(levelname)s] (%(name)s.%(funcName)s) %(message)s")

    # Add a console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Avoid propagating to the root logger
    logger.propagate = False

    return logger
