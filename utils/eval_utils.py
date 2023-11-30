from wandb.apis.public import Run, Api
import os
import hydra
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


def extract_runid(run: Run) -> str:
    """
    Extracts the run id from a W&B run.

    Args:
        run (Run): W&B run object

    Returns:
        run_id (str): Run id of the given run
    """
    return run.id


def extract_config(run: Run) -> dict:
    """
    Extracts the relevant configs that identify an experiment from a W&B run.

    Args:
        run (Run): W&B run object
    """
    config = run.config

    run_id = run.id
    dataset = config["dataset"]["name"]
    method = config["method"]["name"]
    sot = config["sot"]
    n_way = config["n_way"]
    n_shot = config["n_shot"]

    return {
        "run_id": run_id,
        "dataset": dataset,
        "method": method,
        "sot": sot,
        "n_way": n_way,
        "n_shot": n_shot,
    }


def extract_metrics(run: Run) -> dict:
    """
    Extracts the relevant metrics from a W&B run.

    Args:
        run (Run): W&B run object

    Returns:
        metrics (dict): Dictionary of metrics
    """
    return {k: v for k, v in run.summary.items() if not k.startswith("_")}


def load_to_df(runs: list[Run]) -> pd.DataFrame:
    """
    Loads all runs into a pandas DataFrame.

    Args:
        runs (list[Run]): List of W&B runs

    Returns:
        df (pd.DataFrame): DataFrame containing all runs
    """
    configs = [extract_config(run) for run in runs]
    metrics = [extract_metrics(run) for run in runs]

    # Creating joint DataFrame
    df = pd.DataFrame(configs).join(pd.DataFrame(metrics)).set_index("run_id")

    # Creating Multi-Column Index
    column_tuples = [("config", col) for col in df.columns[: len(configs[0])]] + [
        ("eval", col) for col in df.columns[len(configs[0]) :]
    ]
    df.columns = pd.MultiIndex.from_tuples(column_tuples)

    return df


def download_artifact(
    api: Api, wandb_entity: str, wandb_project: str, artifact_dir: str, run_id: str
) -> torch.nn.Module:
    """
    Downloads given artifact from W&B API to the given directory.

    Args:
        api: W&B API
        wandb_entity: W&B entity
        wandb_project: W&B project
        artifact_dir: Directory to store artifact
        run_id: Run id of the model to load
    """
    artifact = api.artifact(f"{wandb_entity}/{wandb_project}/{run_id}:v0")
    path = os.path.join(artifact_dir, run_id)
    artifact.download(root=path)


def init_dataloader(
    cfg: dict, root_dir: str, mode: str = "train"
) -> torch.utils.data.DataLoader:
    """
    Initialize dataloader for given split using the hydra config.

    Args:
        cfg: Hydra config
        mode: Split to use

    Returns:
        Dataloader for given split
    """

    dataset = hydra.utils.instantiate(
        cfg["dataset"]["set_cls"], mode=mode, root=os.path.join(root_dir, "data")
    )
    dataloader = dataset.get_data_loader(**cfg["dataset"]["loader"])

    return dataset, dataloader


def init_model(cfg: dict, dim: int) -> torch.nn.Module:
    """
    Initialize model using the hydra config.

    Args:
        cfg (dict): Hydra config
        dim (int): Dimension of input data

    Returns:
        model (torch.nn.Module) : Model initialized with given config
    """
    backbone = hydra.utils.instantiate(cfg["dataset"]["backbone"], x_dim=dim)
    model = hydra.utils.instantiate(cfg["method"]["cls"], backbone=backbone)

    return model


def init_run(
    run_config: dict, root_dir: str, data_mode: str = "test"
) -> tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.nn.Module]:
    """
    Initialize run by initializing dataloader and model.

    Args:
        run_config (dict): Hydra config
        data_mode (str): Split to use

    Returns:
        dataset (torch.utils.data.Dataset): Dataset for given split
        loader (torch.utils.data.DataLoader): Dataloader for given split
        model (torch.nn.Module): Model initialized with given config
    """

    dataset, loader = init_dataloader(run_config, root_dir, mode=data_mode)
    model = init_model(run_config, dataset.dim)

    return dataset, loader, model


def eval_run(
    model: torch.nn.Module, test_loader: torch.utils.data.DataLoader
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Evaluates the given model on the given dataloader.

    Args:
        model (torch.nn.Module): Model to evaluate
        test_loader (torch.utils.data.DataLoader): Dataloader to use for evaluation

    Returns:
        episodes_results (list): List of tuples containing the ground truth and predictions for each episode
    """

    # Collect the accuracy for each episode
    episodes_results = []

    for x, y in tqdm(test_loader, desc="Evaluating"):
        # Set the number of query samples and classes
        model.set_nquery(x)
        if model.change_way:
            model.set_nway(x)

        # Get ground truth along with mapping from index to encoding
        y_true = y[:, model.n_support :]
        unq = torch.unique(y_true, dim=1).cpu().numpy()[:, 0]
        idx2encoding = {idx: encoding for idx, encoding in enumerate(unq)}
        y_true = y_true.reshape(model.n_way * model.n_query).cpu().numpy()

        # Get predictions and map them from index to encoding
        scores = model.set_forward(x)
        _, topk_labels = scores.data.topk(k=1, dim=1, largest=True, sorted=True)
        y_pred = topk_labels.cpu().numpy()[:, 0]
        y_pred = np.array([idx2encoding[idx] for idx in y_pred])

        # Save to evals
        episodes_results.append((y_true, y_pred))

    return episodes_results


def compute_metrics(
    metric_fns: list[tuple], episodes_results: list[tuple]
) -> pd.DataFrame:
    """
    Evaluates the given model on the given dataloader.

    Args:
        metric_fns (list[tuple]): List of tuples containing the metric function and kwargs
        episodes_results (list[tuple]): List of tuples containing the ground truth and predictions for each episode

    Returns:
        df (pd.DataFrame): DataFrame containing the computed metrics for each episode.
    """

    # Save the evals in a dict
    all_evals = dict()
    for metric_fn, kwargs in metric_fns:
        # Parse kwargs
        kwargs = kwargs if kwargs is not None else {}

        # Get the metric name
        metric_name = " ".join(metric_fn.__name__.capitalize().split("_"))

        # Eval the episodes using the metric fn
        episodes_evals = [
            metric_fn(y_true, y_pred, **kwargs) for y_true, y_pred in episodes_results
        ]

        # Save the evals
        all_evals[metric_fn.__name__] = episodes_evals

    # Create a DataFrame with the evals
    df = pd.DataFrame(all_evals)

    return df
