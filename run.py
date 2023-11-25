import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from prettytable import PrettyTable

from utils.io_utils import (
    check_cfg,
    get_device,
    fix_seed,
    hydra_setup,
    print_cfg,
    get_logger,
)
from utils.train_utils import initialize_dataset_model, train, test


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    # Project setup
    hydra_setup()
    check_cfg(cfg)
    fix_seed(cfg.exp.seed)

    # Print experiment configuration
    logger = get_logger(__name__, cfg)
    logger.info("Starting experiment with configuration:\n")
    print_cfg(cfg)

    # Initialise data loader and model
    device = get_device(cfg.exp.device)
    train_loader, val_loader, model = initialize_dataset_model(cfg, device)

    # Train the model if specified
    if cfg.exp.mode == "train":
        model = train(train_loader, val_loader, model, cfg)

    # Test the model on the specified splits
    results = []
    for split in cfg.eval.eval_split:
        logger.info(f"Testing on {split} split.")
        acc_mean, acc_std = test(cfg, model, split)
        results.append([split, acc_mean, acc_std])

    logger.info(f"Results logged to ./checkpoints/{cfg.exp.name}/results.txt")

    # Display training results (from W&B) in a table
    logger.info("Log training results to W&B.")
    if cfg.exp.mode == "train":
        table = wandb.Table(data=results, columns=["split", "acc_mean", "acc_std"])
        wandb.log({"eval_results": table})

    # Display test results in a table
    logger.info(f"Final test results on {cfg.eval.eval_split} splits:\n")
    display_table = PrettyTable(["split", "acc_mean", "acc_std"])
    for row in results:
        display_table.add_row(row)
    print(display_table)

    wandb.finish()


if __name__ == "__main__":
    main()
