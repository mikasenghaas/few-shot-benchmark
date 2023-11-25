"""
Module containing helper functions to be used in run.py
"""
import glob
import os
import random
import logging

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn


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


def get_exp_name(cfg: OmegaConf) -> str:
    """
    Returns the experiment name to be used for logging and checkpointing.
    """
    name = "{}-{} {}-shot {}-way".format(
        cfg.dataset.name,
        cfg.method.name,
        cfg.exp.n_shot,
        cfg.exp.n_way,
    )

    return name


def hydra_setup() -> None:
    """
    Setup hydra.

    TODO: Figure exactly what this does.
    """
    os.environ["HYDRA_FULL_ERROR"] = "1"
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
    if "name" not in cfg.exp:
        raise ValueError("The 'exp.name' argument is required!")

    if cfg.exp.mode not in ["train", "test"]:
        raise ValueError(f"Unknown mode: {cfg.exp.mode}")


def get_device(device: str | None = None) -> torch.device:
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
    # Turn into dict
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Filter out keys starting with "_"
    cfg = filter_keys(cfg)

    print(OmegaConf.to_yaml(cfg))


def get_logger(name: str, cfg: OmegaConf) -> logging.Logger:
    """
    Create a custom logger with specific formatting.

    Args:
        name: str, name of the logger
        level: logging level

    Returns:
        logger: logging.Logger
    """
    # Return existing logger if exists
    level = logging.getLevelName(cfg.exp.log_level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("[%(levelname)s] (%(name)s.%(funcName)s) %(message)s")

    # Add a console handler if not already present
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Avoid propagating to the root logger
    logger.propagate = False

    return logger
