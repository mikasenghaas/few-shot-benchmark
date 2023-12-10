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
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

import datasets
from utils.io_utils import (
    get_logger,
    model_to_dict,
    opt_to_dict,
)


def initialize_dataset_model(cfg: DictConfig, device: torch.device, root: str = "./"):
    """
    Initialise dataset and model based on config file and returns
    the data loaders and model.

    Args:
        cfg: Hydra config object
        device: torch.device

    Returns:
        train_dataset: datasets.dataset.FewShotDataset
        val_dataset: datasets.dataset.FewShotDataset
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
                root=os.path.join(root, "data"),
                mode="train",
            )
        case "meta":
            logger.info(
                f"Initializing train {cfg.dataset.set_cls._target_}. (Using {(100 * cfg.dataset.subset):.0f}%)"
            )
            train_dataset = instantiate(
                cfg.dataset.set_cls,
                mode="train",
                n_episodes=100,
                root=os.path.join(root, "data"),
            )
        case _:
            raise ValueError(f"Unknown method type: {cfg.method.type}")

    # Instantiate val dataset (few-shot)
    logger.info(
        f"Initializing val {cfg.dataset.set_cls._target_}. (Using {(100 * cfg.dataset.subset):.0f}%)"
    )
    val_dataset = instantiate(
        cfg.dataset.set_cls, mode="val", n_episodes=100, root=os.path.join(root, "data")
    )
    # Instantiate test dataset (few-shot) to test if we have enough classes in it
    # with at least (n_support + n_query) examples to form n_way
    logger.info(
        f"Initializing test {cfg.dataset.set_cls._target_}. (Using {(100 * cfg.dataset.subset):.0f}%)"
    )
    test_dataset = instantiate(
        cfg.dataset.set_cls,
        n_episodes=100,
        mode="test",
        root=os.path.join(root, "data"),
    )

    # Initialise SOT (if specified)
    sot = None
    if cfg.use_sot:
        logger.info("Initialising SOT")
        # Transformed feature dim is batch size which for episodic training is: n_way * (n_support + n_query)
        final_feat_dim = cfg.n_way * (cfg.n_shot + cfg.n_query)
        sot = instantiate(cfg.sot.cls, final_feat_dim=final_feat_dim)

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
    model = instantiate(cfg.method.cls, backbone=backbone, sot=sot)
    model = model.to(device)

    if cfg.method.name == "maml":
        cfg.train.max_epochs *= model.n_task  # maml use multiple tasks in one update

    return train_dataset, val_dataset, test_dataset, model


def train(
    train_dataset: datasets.cell.tabula_muris.TMSetDataset
    | datasets.prot.swissprot.SPSetDataset,
    val_dataset: datasets.cell.tabula_muris.TMSetDataset
    | datasets.prot.swissprot.SPSetDataset,
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
        train_dataset: datasets.cell.tabula_muris.TMSetDataset | datasets.prot.swissprot.SPSetDataset
        val_dataset: datasets.cell.tabula_muris.TMSetDataset | datasets.prot.swissprot.SPSetDataset
        model: torch.nn.Module
        cfg: Hydra config object

    Returns:
        model: torch.nn.Module
    """
    logger = get_logger(__name__, cfg)

    # Get the train and val loaders
    train_loader = train_dataset.get_data_loader(
        num_workers=cfg.dataset.loader.num_workers,
        pin_memory=cfg.dataset.loader.pin_memory,
        episodes=100,
    )
    val_loader = val_dataset.get_data_loader(
        num_workers=cfg.dataset.loader.num_workers,
        pin_memory=cfg.dataset.loader.pin_memory,
        episodes=100,
    )

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
    max_acc = -1
    patience = cfg.train.patience  # the number of epochs to wait before early stop
    if patience % cfg.general.val_freq != 0:
        raise ValueError(
            f"Patience ({patience}) must be divisible by validation frequency ({cfg.general.val_freq})"
        )
    epochs_since_improvement = 0
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
                outfile = os.path.join(cfg.paths.log_dir, "best_model.pt")
                torch.save(model.state_dict(), outfile)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += cfg.general.val_freq

            if epochs_since_improvement >= patience:
                logger.info(
                    f"Early stopping triggered in epoch {epoch + 1} because "
                    f"val/acc hasn't improved for {epochs_since_improvement} epochs."
                )
                break

    # Log best model to W&B
    model_artifact.add_dir(cfg.paths.log_dir)
    wandb.log_artifact(model_artifact)

    # Load best model
    model.load_state_dict(torch.load(outfile))

    return model


def test(
    cfg: DictConfig,
    model: nn.Module,
    dataset: datasets.cell.tabula_muris.TMSetDataset
    | datasets.prot.swissprot.SPSetDataset,
    split: str,
):
    """
    Test loop. Loads model from checkpoint and evaluates on test data.
    Writes results to file in checkpoint directory.

    Args:
        cfg: Hydra config object
        model: torch.nn.Module
        dataset: datasets.cell.tabula_muris.TMSetDataset | datasets.prot.swissprot.SPSetDataset
        split: str, one of ["val", "test"]

    Returns:
        acc_mean: float
        acc_ci: float
        acc_std: float
    """
    # instantiate train dataset again, but this time as set_dataset if method is baseline
    if cfg.method.type == "baseline":
        dataset = instantiate(
            cfg.dataset.set_cls,
            mode=split,
            n_episodes=cfg.eval.n_episodes,
        )

    # Get the test loader
    test_loader = dataset.get_data_loader(
        num_workers=cfg.dataset.loader.num_workers,
        pin_memory=cfg.dataset.loader.pin_memory,
        episodes=cfg.eval.n_episodes,
    )

    if next(iter(test_loader))[0].shape[0] < cfg.n_way:
        message = (
            f"there are not enough classes in {split} split\n"
            f"to form {cfg.n_way} way test (max {next(iter(test_loader))[0].shape[0]} way test)\n"
            f"try reducing n_support ({cfg.n_shot}) or n_query ({cfg.exp.n_query})"
        )
        raise ValueError(message)

    # Test loop
    model.eval()
    acc_mean, acc_ci, acc_std = model.test_loop(test_loader)

    return acc_mean, acc_ci, acc_std
