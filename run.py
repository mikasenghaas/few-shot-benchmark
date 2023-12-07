import hydra
from omegaconf import DictConfig
import wandb
from prettytable import PrettyTable

import rootutils

from utils.io_utils import (
    check_cfg,
    get_device,
    fix_seed,
    hydra_setup,
    get_logger,
    print_cfg,
)
from utils.train_utils import initialize_dataset_model, train, test

# Setup root environment
root_path = rootutils.setup_root(__file__)
rootutils.set_root(
    path=root_path,
    project_root_env_var=True,
)


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    # Project setup
    hydra_setup(cfg)
    check_cfg(cfg)
    fix_seed(cfg.general.seed)

    # Print experiment configuration
    logger = get_logger(__name__, cfg)
    print_cfg(cfg)

    # Initialise data loader and model
    device = get_device(cfg.general.device)
    train_dataset, val_dataset, test_dataset, model = initialize_dataset_model(
        cfg, device
    )

    # Train the model if specified
    best_model = train(train_dataset, val_dataset, model, cfg)

    # Test the model on the specified splits
    display_table = PrettyTable(["split", "acc_mean", "acc_std"])
    for split in cfg.eval.splits:
        logger.info(f"Testing on {split} split.")
        dataset = (
            test_dataset
            if split == "test"
            else (val_dataset if split == "val" else train_dataset)
        )
        acc_mean, acc_ci, acc_std = test(
            cfg,
            best_model,
            dataset,
            split,
        )
        display_table.add_row(
            [split, f"{acc_mean:.2f} ± {acc_ci:.2f}%", f"{acc_std:.2f}"]
        )
        wandb.log(
            {
                f"{split}/acc": acc_mean,
                f"{split}/acc_std": acc_std,
                f"{split}/acc_ci": acc_ci,
            }
        )

    print(display_table)

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
