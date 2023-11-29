import hydra
from omegaconf import DictConfig, OmegaConf
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
    train_loader, val_loader, model = initialize_dataset_model(cfg, device)

    # Train the model if specified
    best_model = train(train_loader, val_loader, model, cfg)

    # Test the model on the specified splits
    results = []
    for split in cfg.eval.splits:
        logger.info(f"Testing on {split} split.")
        acc_mean, acc_std = test(cfg, best_model, split)
        results.append([split, acc_mean, acc_std])

    # Display training results (from W&B) in a table
    logger.info("Log training results to W&B.")
    table = wandb.Table(data=results, columns=["split", "acc_mean", "acc_std"])
    wandb.log({"eval_results": table})

    # Display test results in a table
    logger.info(f"Final test results on {cfg.eval.splits} splits:\n")
    display_table = PrettyTable(["split", "acc_mean", "acc_std"])
    for row in results:
        display_table.add_row(row)
    print(display_table)

    wandb.finish()


if __name__ == "__main__":
    main()
