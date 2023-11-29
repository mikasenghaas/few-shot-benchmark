"""
Module containing functions for loading datasets and models, and 
training and evaluation loops.

Includes:
    - initialize_dataset_model: Instantiate dataset and model based on Hydra config.
    - train: Training loop.
    - test: Evaluation loop.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

from utils.io_utils import (
    get_logger,
    model_to_dict,
    opt_to_dict,
)


def initialize_dataset_model(cfg: DictConfig, device: torch.device):
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
    match cfg.method.type:
        case "baseline":
            logger.info(
                f"Initializing train {cfg.dataset.simple_cls._target_}. (Using {(100 * cfg.dataset.subset):.0f}%)"
            )
            train_dataset = instantiate(
                cfg.dataset.simple_cls,
                batch_size=cfg.train.batch_size,
                mode="train",
            )
        case "meta":
            logger.info(
                f"Initializing train {cfg.dataset.set_cls._target_}. (Using {(100 * cfg.dataset.subset):.0f}%)"
            )
            train_dataset = instantiate(cfg.dataset.set_cls, mode="train")
        case _:
            raise ValueError(f"Unknown method type: {cfg.method.type}")

    # Instantiate val dataset
    match cfg.eval.type:
        case "simple":
            logger.info(
                f"Initializing val {cfg.dataset.simple_cls._target_}. (Using {(100 * cfg.dataset.subset):.0f}%)"
            )
            val_dataset = instantiate(
                cfg.dataset.simple_cls,
                batch_size=cfg.train.val_batch,
                mode="val",
            )
        case _:
            logger.info(
                f"Initializing val {cfg.dataset.set_cls._target_}. (Using {(100 * cfg.dataset.subset):.0f}%)"
            )
            val_dataset = instantiate(cfg.dataset.set_cls, mode="val")

    # Instantiate backbone (For MAML, need to instantiate backbone with fast weight)
    logger.info(f"Initialise backbone {cfg.dataset.backbone._target_}")
    if cfg.method.fast_weight:
        backbone = instantiate(
            cfg.dataset.backbone, x_dim=train_dataset.dim, fast_weight=True
        )
    else:
        backbone = instantiate(cfg.dataset.backbone, x_dim=train_dataset.dim)

    # Instatiante model with backbone
    logger.info(f"Initialise method {cfg.method.cls._target_}")
    model = instantiate(cfg.method.cls, backbone=backbone)
    model = model.to(device)

    # Get train and val data loaders
    train_loader = train_dataset.get_data_loader(
        num_workers=cfg.dataset.loader.num_workers,
        pin_memory=cfg.dataset.loader.pin_memory,
    )
    val_loader = val_dataset.get_data_loader(
        num_workers=cfg.dataset.loader.num_workers,
        pin_memory=cfg.dataset.loader.pin_memory,
    )

    if cfg.method.name == "maml":
        cfg.train.max_epochs *= model.n_task  # maml use multiple tasks in one update

    return train_loader, val_loader, model


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    cfg: DictConfig,
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

    # Initialise W&B run
    logger.info("Initializing W&B")
    run = wandb.init(
        name=cfg.name,
        group=cfg.group,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        settings=wandb.Settings(start_method="thread"),
        dir=cfg.paths.log_dir,
        mode=cfg.wandb.mode,
    )
    wandb.define_metric("*", step_metric="epoch")

    # Instantiate optimizer
    logger.info(f"Initialise Adam with lr {cfg.train.lr}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # Log model and optimizer details to W&B
    logger.info("Log model and optimiser details to W&B")
    wandb.config.update({"model_details": model_to_dict(model)})
    wandb.config.update({"optimizer_details": opt_to_dict(optimizer)})

    # Initialise W&B artifact for model
    model_artifact = wandb.Artifact(name=run.id, type="model")

    # Training loop
    best_model = None
    max_acc = -1
    logger.info("Start training")
    for epoch in range(cfg.train.max_epochs):
        wandb.log({"epoch": epoch + 1})
        model.train()
        loss = model.train_loop(epoch, train_loader, optimizer)
        wandb.log({"train/loss": loss})

        # Validation loop on every val_freq or last epoch
        if epoch % cfg.general.val_freq == 0 or epoch == cfg.train.max_epochs - 1:
            model.eval()
            acc, acc_ci, acc_std = model.test_loop(val_loader)
            wandb.log({"val/acc": acc, "val/acc_ci": acc_ci, "val/acc_std": acc_std})

            if acc > max_acc:
                logger.info(f"New best model! (Acc. {acc:.3f} > {max_acc:.3f})")
                max_acc = acc
                best_model = model
                outfile = os.path.join(cfg.paths.log_dir, "best_model.pt")
                torch.save(model.state_dict(), outfile)

    # Log best model to W&B
    model_artifact.add_dir(cfg.paths.log_dir)
    wandb.log_artifact(model_artifact)

    return best_model


def test(cfg: DictConfig, model: nn.Module, split: str):
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

    # Instantiate test dataset
    logger.info(
        f"Initialise {split} {cfg.dataset.name} dataset with {cfg.eval.n_episodes} episodes"
    )
    test_dataset = instantiate(
        cfg.dataset.set_cls, n_episodes=cfg.eval.n_episodes, mode=split
    )

    # Get the test loader
    test_loader = test_dataset.get_data_loader(
        num_workers=cfg.dataset.loader.num_workers,
        pin_memory=cfg.dataset.loader.pin_memory,
    )

    # Test loop
    model.eval()
    acc_mean, acc_ci, acc_std = model.test_loop(test_loader)

    return acc_mean, acc_ci, acc_std
