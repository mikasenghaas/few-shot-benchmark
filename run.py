import hydra
from omegaconf import OmegaConf
import wandb
from prettytable import PrettyTable

from utils.train_utils import initialize_dataset_model, train, test
from utils.io_utils import (
    check_cfg,
    get_device,
    fix_seed,
    hydra_setup,
    print_cfg,
)


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg):
    # Project setup
    hydra_setup()
    check_cfg(cfg)
    fix_seed(cfg.exp.seed)

    # Print experiment configuration
    print_cfg(cfg)

    # Initialise data loader and model
    device = get_device(cfg.device)
    train_loader, val_loader, model = initialize_dataset_model(cfg, device)

    # Train the model if specified
    if cfg.mode == "train":
        model = train(train_loader, val_loader, model, cfg)

    # Test the model on the specified splits
    results = []
    # print("Checkpoint directory:", cfg.checkpoint.dir)
    for split in cfg.eval_split:
        acc_mean, acc_std = test(cfg, model, split)
        results.append([split, acc_mean, acc_std])

    # print(f"Results logged to ./checkpoints/{cfg.exp.name}/results.txt")

    # Display training results (from W&B) in a table
    if cfg.mode == "train":
        table = wandb.Table(data=results, columns=["split", "acc_mean", "acc_std"])
        wandb.log({"eval_results": table})

    # Display test results in a table
    display_table = PrettyTable(["split", "acc_mean", "acc_std"])
    for row in results:
        display_table.add_row(row)

    print(display_table)

    wandb.finish()


if __name__ == "__main__":
    main()
