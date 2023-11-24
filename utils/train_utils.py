"""
Module containting functions for loading datasets and models, and 
training and evaluation loops.

Includes:
    - initialize_dataset_model: Instantiate dataset and model based on Hydra config.
    - train: Training loop.
    - test: Evaluation loop.
"""

import time
import os
import math
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils.io_utils import (
    get_logger,
    get_resume_file,
    model_to_dict,
    opt_to_dict,
    get_model_file,
)


def initialize_dataset_model(cfg: OmegaConf, device: torch.device):
    """
    Initialise dataset and model based on config file and returns
    the data loaders and model.

    Args:
        cfg: Hydra config object
        device: torch.device

    Returns:
        train_loader: torch.utils.data.DataLoader
        val_loader: torch.utils.data.DataLoader
        model: torch.nn.Module
    """
    logger = get_logger(__name__, cfg)

    # Instatiate train dataset
    logger.info("Initializing training dataset.")
    match cfg.method.type:
        case "baseline":
            train_dataset = instantiate(
                cfg.dataset.simple_cls, batch_size=cfg.method.train_batch, mode="train"
            )
        case "meta":
            train_dataset = instantiate(cfg.dataset.set_cls, mode="train")
        case _:
            raise ValueError(f"Unknown method type: {cfg.method.type}")

    # Instantiate val dataset
    logger.info("Initializing validation dataset.")
    match cfg.method.eval_type:
        case "simple":
            val_dataset = instantiate(
                cfg.dataset.simple_cls, batch_size=cfg.method.val_batch, mode="val"
            )
        case _:
            val_dataset = instantiate(cfg.dataset.set_cls, mode="val")

    # Instantiate backbone (For MAML, need to instantiate backbone with fast weight)
    if cfg.method.fast_weight:
        logger.info("Initialise backbone (with fast weight)")
        backbone = instantiate(cfg.backbone, x_dim=train_dataset.dim, fast_weight=True)
    else:
        logger.info("Initialise backbone (no fast weight)")
        backbone = instantiate(cfg.backbone, x_dim=train_dataset.dim)

    # Instatiante model with backbone
    logger.info("Initialise model")
    model = instantiate(cfg.method.cls, backbone=backbone)
    model = model.to(device)

    # Get train and val data loaders
    train_loader = train_dataset.get_data_loader()
    val_loader = val_dataset.get_data_loader()

    if cfg.method.name == "maml":
        cfg.method.stop_epoch *= model.n_task  # maml use multiple tasks in one update

    return train_loader, val_loader, model


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    cfg: OmegaConf,
):
    """
    Full training loop over epochs. Saves model checkpoints and logs to wandb.
    Validates every val_freq epochs on the validation data, and saves the model
    every save_freq epochs. Supports resuming training from the most recent
    checkpoint for a unique combination of experiment name, dataset, method,
    and model.

    Args:
        train_loader: torch.utils.data.DataLoader
        val_loader: torch.utils.data.DataLoader
        model: torch.nn.Module
        cfg: Hydra config object

    Returns:
        model: torch.nn.Module
    """
    logger = get_logger(__name__, cfg)
    logger.info("Starting model training")

    # Set checkpoint directory (based on combination of experiment name, dataset, method, model and time)
    cfg.checkpoint.time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    cp_dir = os.path.join(cfg.checkpoint.dir, cfg.checkpoint.time)

    # Create checkpoint directory if it doesn't exist
    if not os.path.isdir(cp_dir):
        os.makedirs(cp_dir)

    # Create name for W&B run
    name = "%s-%s-%s %sshot %sway" % (
        cfg.dataset.name,
        cfg.model,  # see main.yaml, this setting doesn't change the model, TODO figure out how to change it (changing dataset.backbone._target_ introduces errors)
        cfg.method.name,
        cfg.n_shot,
        cfg.n_way,
    )

    # Initialise W&B run
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
        group=cfg.exp.name,
        settings=wandb.Settings(start_method="thread"),
        mode=cfg.wandb.mode,
    )
    wandb.define_metric("*", step_metric="epoch")

    # Resume training if specified
    if cfg.exp.resume:
        resume_file = get_resume_file(cp_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            cfg.method.start_epoch = tmp["epoch"] + 1
            model.load_state_dict(tmp["state"])

    # Instantiate optimizer
    optimizer = instantiate(cfg.optimizer_cls, params=model.parameters())

    # Log model and optimizer details to W&B
    wandb.config.update({"model_details": model_to_dict(model)})
    wandb.config.update({"optimizer_details": opt_to_dict(optimizer)})

    # Training loop
    max_acc = -1
    for epoch in range(cfg.method.start_epoch, cfg.method.stop_epoch):
        wandb.log({"epoch": epoch})
        model.train()
        model.train_loop(epoch, train_loader, optimizer)

        # Validation loop on every val_freq or last epoch
        if epoch % cfg.exp.val_freq == 0 or epoch == cfg.method.stop_epoch - 1:
            model.eval()
            acc = model.test_loop(val_loader)
            # print(f"Epoch {epoch}: {acc:.2f}")
            wandb.log({"acc/val": acc})

            if acc > max_acc:
                # print("best model! save...")
                max_acc = acc
                outfile = os.path.join(cp_dir, "best_model.tar")
                torch.save({"epoch": epoch, "state": model.state_dict()}, outfile)

        # Save model on every save_freq or last epoch
        if epoch % cfg.exp.save_freq == 0 or epoch == cfg.method.stop_epoch - 1:
            outfile = os.path.join(cp_dir, "{:d}.tar".format(epoch))
            torch.save({"epoch": epoch, "state": model.state_dict()}, outfile)

    return model


def test(cfg: OmegaConf, model: nn.Module, split: str):
    """
    Test loop. Loads model from checkpoint and evaluates on test data.
    Writes results to file in checkpoint directory.

    Args:
        cfg: Hydra config object
        model: torch.nn.Module
        split: str, one of ["val", "test"]

    Returns:
        acc_mean: float
        acc_std: float
    """
    logger = get_logger(__name__, cfg)
    logger.info("Starting model testing")

    # Instantiate test dataset
    match cfg.method.type:
        case "simple":
            test_dataset = instantiate(
                cfg.dataset.simple_cls, batch_size=cfg.method.val_batch, mode=split
            )
        case _:
            test_dataset = instantiate(
                cfg.dataset.set_cls, n_episode=cfg.iter_num, mode=split
            )

    # Get the test loader
    test_loader = test_dataset.get_data_loader()

    # Load model from checkpoint (either latest or specified)
    model_file_path = get_model_file(cfg)
    model.load_state_dict(torch.load(model_file_path)["state"])
    model.eval()

    # Test loop
    match cfg.method.eval_type:
        case "simple":
            acc_all = []

            num_iters = math.ceil(cfg.iter_num / len(test_dataset.get_data_loader()))
            cfg.iter_num = num_iters * len(test_dataset.get_data_loader())
            # print("num_iters", num_iters)
            for i in range(num_iters):
                acc_mean, acc_std = model.test_loop(test_loader, return_std=True)
                acc_all.append(acc_mean)

            acc_mean = np.mean(acc_all)
            acc_std = np.std(acc_all)
        case _:
            acc_mean, acc_std = model.test_loop(test_loader, return_std=True)

    # Write results to file in checkpoint directory
    with open(f"./checkpoints/{cfg.exp.name}/results.txt", "a") as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        exp_setting = "%s-%s-%s-%s %sshot %sway" % (
            cfg.dataset.name,
            split,
            cfg.model,
            cfg.method.name,
            cfg.n_shot,
            cfg.n_way,
        )

        acc_str = "%4.2f%% +- %4.2f%%" % (
            acc_mean,
            1.96 * acc_std / np.sqrt(cfg.iter_num),
        )
        f.write(
            "Time: %s, Setting: %s, Acc: %s, Model: %s \n"
            % (timestamp, exp_setting, acc_str, model_file_path)
        )

    return acc_mean, acc_std
