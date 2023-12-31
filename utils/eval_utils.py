from omegaconf import OmegaConf
from wandb.apis.public import Run, Api
import os
import hydra
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import sys

sys.path.append("..")
import utils.train_utils as train_utils  # noqa: E402


def extract_runid(run: Run) -> str:
    """
    Extracts the run id from a W&B run.

    Args:
        run (Run): W&B run object

    Returns:
        run_id (str): Run id of the given run
    """
    return run.id


def extract_info(run: Run) -> dict:
    """
    Extracts the relevant info from a W&B run.

    Args:
        run (Run): W&B run object
    """
    info = {
        "id": run.id,
        "name": run.name,
        "runtime": run.summary["_runtime"],
    }

    return info


def extract_config(run: Run) -> dict:
    """
    Extracts the relevant configs that identify an experiment from a W&B run.

    Args:
        run (Run): W&B run object
    """
    config = run.config

    dataset = config["dataset"]["name"]
    method = config["method"]["name"]
    use_sot = config["use_sot"]
    n_way = config["n_way"]
    n_shot = config["n_shot"]

    return {
        "dataset": dataset,
        "method": method,
        "use_sot": use_sot,
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


def extract_hyperparams(run: Run) -> dict:
    """
    Extracts the relevant hyperparameters from a W&B run.

    Args:
        run (Run): W&B run object

    Returns:
        hyperparams (dict): Dictionary of hyperparameters
    """
    config = run.config

    lr = config["train"]["lr"]
    feat_dim = config["dataset"]["backbone"]["feat_dim"]
    sot_reg = config["sot"]["cls"]["ot_reg"]
    sot_dist_metric = config["sot"]["cls"]["distance_metric"]

    return {
        "lr": lr,
        "feat_dim": feat_dim,
        "sot_reg": sot_reg,
        "sot_dist_metric": sot_dist_metric,
    }


def load_to_df(runs: list[Run]) -> pd.DataFrame:
    """
    Loads all runs into a pandas DataFrame.

    Args:
        runs (list[Run]): List of W&B runs

    Returns:
        df (pd.DataFrame): DataFrame containing all runs
    """
    info = [extract_info(run) for run in runs]
    configs = [extract_config(run) for run in runs]
    metrics = [extract_metrics(run) for run in runs]
    hparams = [extract_hyperparams(run) for run in runs]

    # Creating joint DataFrame
    df = (
        pd.DataFrame(info)
        .join(pd.DataFrame(configs))
        .join(pd.DataFrame(metrics))
        .join(pd.DataFrame(hparams))
        .set_index("id")
    )

    # Creating Multi-Column Index
    info_columns = [("info", col) for col in info[0].keys() if col != "id"]
    config_columns = [("config", col) for col in configs[0].keys()]
    eval_columns = [("eval", col) for col in metrics[0].keys()]
    hparams_columns = [("hparams", col) for col in hparams[0].keys()]
    column_tuples = info_columns + config_columns + eval_columns + hparams_columns
    df.columns = pd.MultiIndex.from_tuples(column_tuples)

    return df


def get_best_run(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Filters the given DataFrame for the best run according to the given metric
    for each unique experiment configuration (in the column keys (config, *)).

    Args:
        df (pd.DataFrame): DataFrame containing all runs
        metric (str): Metric to use for filtering

    Returns:
        df (pd.DataFrame): DataFrame containing the best run for each experiment
        configuration
    """
    # Get all columns containing the experiment configuration
    experiment_config = list(filter(lambda x: x[0] == "config", df.columns))

    # Get all unique experiment configurations
    unique_experiments = df[experiment_config].drop_duplicates()

    # Get best runs for each experiment configuration
    best_runs = []
    for _, experiment in unique_experiments.iterrows():
        # Get all runs with the same experiment configuration
        experiment_runs = df[
            (df[experiment_config] == experiment[experiment_config]).all(axis=1)
        ]

        # Get run with best validation accuracy
        best_run = experiment_runs.sort_values(metric, ascending=False).iloc[0]
        best_runs.append(best_run)

    return pd.DataFrame(best_runs)


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
    run, root_dir: str, data_mode: str = "test"
) -> tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.nn.Module]:
    """
    Initialize run by initializing dataloader and model.

    Args:
        run (wandb.Run): W&B run
        data_mode (str): Split to use

    Returns:
        dataset (torch.utils.data.Dataset): Dataset for given split
        loader (torch.utils.data.DataLoader): Dataloader for given split
        model (torch.nn.Module): Model initialized with given config
    """
    # Extract the hydra config
    config = run.config

    # Initialise dataset and model
    dataset, loader = init_dataloader(config, root_dir, mode=data_mode)
    model = init_model(config, dataset.dim)

    return dataset, loader, model


def init_all(
    run, root_dir: str = "../"
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.nn.Module,
]:
    """
    Initialize run by initializing dataloader and model.

    Args:
        run (wandb.Run): W&B run
        data_mode (str): Split to use

    Returns:
        dataset (torch.utils.data.Dataset): Dataset for given split
        loader (torch.utils.data.DataLoader): Dataloader for given split
        model (torch.nn.Module): Model initialized with given config
    """
    run_config = OmegaConf.create(run.config)
    train_data, val_data, test_data, model = train_utils.initialize_dataset_model(
        run_config, device="cpu", root=root_dir
    )

    train_loader = train_data.get_data_loader(**run.config["dataset"]["loader"])
    eval_loader = val_data.get_data_loader(**run.config["dataset"]["loader"])
    test_loader = test_data.get_data_loader(**run.config["dataset"]["loader"])

    return train_loader, eval_loader, test_loader, model


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


def visualise_episode(
    loader: torch.utils.data.DataLoader,
    model: nn.Module | None = None,
    ax: plt.Axes | None = None,
):
    """
    Visualise an episode of a few-shot learning dataset.

    Args:
        loader (torch.data.utils.Dataloader): An episodic data loader
        model (torch.nn.Module): Model to evaluate
        embedding (str): Embedding to use for visualisation (one of `input`, `backbone`, `lstm`)
        ax (plt.Axes): Axes to use for plotting
    """

    # Initialise transformers
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    encoder = LabelEncoder()

    # Get the episode data
    x, y = next(iter(loader))

    # Get the embeddings
    outputs = model.set_forward(x, return_intermediates=True)
    c, t = model.correct(x)
    del outputs["scores"]

    # Get the episode parameters
    n_way = model.n_way
    n_support = model.n_support
    n_query = model.n_query

    fig, axs = plt.subplots(ncols=len(outputs), figsize=(5 * len(outputs), 5))
    fig.suptitle(f"Few-shot Accuracy: {100*(c / t):.2f}")

    for ax, (layer, x) in zip(axs, outputs.items()):
        xf = model.reshape2feature(x).detach().numpy()
        yf = model.reshape2feature(y).detach().numpy()

        # PCA transform features and integer-encode labels
        xt = pca.fit_transform(scaler.fit_transform(xf))
        yt = encoder.fit_transform(yf)

        # To tensor
        x, y = torch.Tensor(xt), torch.Tensor(yt)

        # Reshape to set
        x = model.reshape2set(x)
        y = model.reshape2set(y)

        # Split into support and query sets
        xs, xq = model.parse_feature(x)
        ys, yq = model.parse_feature(y)

        xs = xs.reshape(n_way, n_support, -1)
        xq = xq.reshape(n_way, n_query, -1)

        # Compute prototype
        proto = xs.mean(1)
        proto_c = y[:, 0]

        # Re-flatten
        xs, ys = xs.reshape(-1, 2), ys.reshape(-1)
        xq, yq = xq.reshape(-1, 2), yq.reshape(-1)

        # Plot the data
        ax.scatter(
            proto[:, 0],
            proto[:, 1],
            c=proto_c,
            cmap="brg",
            s=200,
            alpha=0.75,
            marker="*",
            label="Prototype",
        )
        ax.scatter(
            xs[:, 0],
            xs[:, 1],
            c=ys,
            cmap="brg",
            s=100,
            alpha=0.05,
            marker="*",
            label="Support",
        )
        ax.scatter(
            xq[:, 0],
            xq[:, 1],
            c=yq,
            cmap="brg",
            s=100,
            alpha=0.5,
            marker="o",
            label="Query",
        )

        ax.set_title(layer)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def visualise_transport_plan(x, model: nn.Module, ax: plt.Axes | None = None):
    if not ax:
        _, ax = plt.subplots(figsize=(10, 10))

    # Get few-shot episode

    # Get the model output (including the transport plan)
    outputs = model.set_forward(x, return_intermediates=True)
    sot_mat = model.reshape2feature(outputs["sot"]).detach().numpy()

    # Compute loss and accuracy
    num_correct, num_total = model.correct(x)
    acc = num_correct / num_total
    # loss = model.set_forward_loss(x)

    # mean_row_sum = np.sum(sot_mat, axis=1).mean()
    # mean_col_sum = np.sum(sot_mat, axis=0).mean()
    sns.heatmap(sot_mat, ax=ax)
    ax.set_title(f"SOT Embeddings (Acc. {100*acc:.2f}%)")
    ax.set_xticks([])
    ax.set_yticks([])


def visualise_lstm(x: torch.Tensor, model: nn.Module, ax: plt.Axes | None = None):
    if not ax:
        _, ax = plt.subplots(figsize=(10, 10))

    # Get the model output (including the transport plan)
    outputs = model.set_forward(x, return_intermediates=True)
    sot_mat = model.reshape2feature(outputs["lstm"]).detach().numpy()

    # Compute loss and accuracy
    num_correct, num_total = model.correct(x)
    acc = num_correct / num_total
    # loss = model.set_forward_loss(x)

    # mean_row_sum = np.sum(sot_mat, axis=1).mean()
    # mean_col_sum = np.sum(sot_mat, axis=0).mean()
    sns.heatmap(sot_mat, ax=ax)
    ax.set_title(f"LSTM Embeddings (Acc. {100*acc:.2f}%)")
    ax.set_xticks([])
    ax.set_yticks([])


def visualise_confusion_matrix(loader, model, ax=None):
    if not ax:
        _, ax = plt.subplots(figsize=(10, 10))

    # Get few-shot episode
    x, _ = next(iter(loader))

    # Get the model output
    outputs = model.set_forward(x)
    logits = outputs["scores"].detach().numpy()
    preds = logits.argmax(axis=1)
    y = model.get_episode_labels(model.n_query).detach().numpy()

    acc = (y == preds).mean()

    conf_matrix = confusion_matrix(y, preds)

    sns.heatmap(conf_matrix, annot=True, ax=ax)
    ax.set_title(f"Confusion Matrix (Acc. {100*acc:.2f}%)")
    ax.set_xticks([])
    ax.set_yticks([])


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
        # metric_name = " ".join(metric_fn.__name__.capitalize().split("_"))

        # Eval the episodes using the metric fn
        episodes_evals = [
            metric_fn(y_true, y_pred, **kwargs) for y_true, y_pred in episodes_results
        ]

        # Save the evals
        all_evals[metric_fn.__name__] = episodes_evals

    # Create a DataFrame with the evals
    df = pd.DataFrame(all_evals)

    return df


def exp2results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the results of the experiments from the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing all runs

    Returns:
        df_results (pd.DataFrame): DataFrame containing the results of the experiments
    """

    # Sort the values by method
    df = df.sort_values(by=[("config", "method")])

    # Get the test accuracy for each method w/ and w/o SOT
    sot_test_acc = df[df[("config", "use_sot")]][("eval", "test/acc")].values
    test_acc = df[~df[("config", "use_sot")]][("eval", "test/acc")].values

    # Get index of the best run for each method w/ and w/o SOT
    sot_best_run_idx = df[df[("config", "use_sot")]][
        ("eval", "test/acc")
    ].values.argmax()
    best_run_idx = df[~df[("config", "use_sot")]][("eval", "test/acc")].values.argmax()

    # Get the confidence intervals
    sot_test_acc_ci = df[df[("config", "use_sot")]][("eval", "test/acc_ci")].values
    test_acc_ci = df[~df[("config", "use_sot")]][("eval", "test/acc_ci")].values

    # Get the methods name and remap them to the styled names
    remmaping = {
        "baseline": "B",
        "baseline_pp": "B++",
        "matchingnet": "MT",
        "protonet": "PT",
        "maml": "MAML",
    }
    methods = sorted(df[("config", "method")].unique())
    methods = [remmaping[method] for method in methods]

    df_results = pd.DataFrame(
        {
            "Method": methods,
            "w/o SOT": [
                f"${acc:.1f} \pm {ci:.1f}$"
                if i != best_run_idx
                else "$\mathbf{" + f"{acc:.1f} \pm {ci:.1f}" + "}$"
                for i, (acc, ci) in enumerate(zip(test_acc, test_acc_ci))
            ],
            "w/ SOT": [
                f"${acc:.1f} \pm {ci:.1f}$"
                if i != sot_best_run_idx
                else "$\mathbf{" + f"{acc:.1f} \pm {ci:.1f}" + "}$"
                for i, (acc, ci) in enumerate(zip(sot_test_acc, sot_test_acc_ci))
            ],
            "Diff": sot_test_acc - test_acc,
        }
    )

    # Round the results
    df_results = df_results.round(2)

    return df_results


def exp2latex(df: pd.DataFrame) -> str:
    """
    Converts the given DataFrame to a latex table.

    Args:
        df (pd.DataFrame): DataFrame containing all runs

    Returns:
        latex (str): Latex table
    """

    # Style the results
    df_styled = (
        df.style.format(precision=1)
        .applymap(lambda x: "font-weight: bold" if x > 0 else "", subset=["Diff"])
        .applymap(lambda x: "color: red" if x < 0 else "color: teal", subset=["Diff"])
    )

    # Define the caption
    caption = "\\textbf{Benchmark Results}. Test accuracy of all methods on TM and SP \
    \ in the 5-way-5-shot setting. We depict the average accuracy and the 95\% confidence  \
    \ interval both without (left) and with SOT (right) and the difference."

    # Convert to latex
    latex = df_styled.to_latex(
        position="h",
        hrules=True,
        clines=None,
        label="tab:tuned-benchmark",
        caption=caption,
        sparse_index=True,
        multirow_align="c",
        convert_css=True,
    )

    # Add Midrule between dataset tables
    search_term = r"\multirow[c]{5}{*}{SP}"
    index_to_insert_midrule = latex.find(search_term)
    if index_to_insert_midrule != -1:
        latex = (
            latex[:index_to_insert_midrule]
            + "\\midrule\n"
            + latex[index_to_insert_midrule:]
        )
    else:
        print("❌ Could not find the row to insert the midrule.")

    # Change Diff to Diff (%)
    latex = latex.replace("Diff", "Diff (\%)")

    # Erase the row where "Method" is
    search_term = r" & Method &  &  &  \\"
    index_to_erase = latex.find(search_term)
    if index_to_erase != -1:
        latex = (
            latex[:index_to_erase]
            + latex[index_to_erase:].replace(search_term, "").strip()
        )
    else:
        print("❌ Could not find the row to erase.")

    # Add centering to the table
    search_term = r"\begin{tabular}"
    index_to_insert_centering = latex.find(search_term)
    if index_to_insert_centering != -1:
        latex = (
            latex[:index_to_insert_centering]
            + "\\centering\n"
            + latex[index_to_insert_centering:]
        )
    else:
        print("❌ Could not find the row to insert centering.")

    # Insert test accuracy to the top of the table
    search_term = r"\toprule"
    top_rule_index = latex.find(search_term) + len(search_term)
    if top_rule_index != -1:
        latex = (
            latex[:top_rule_index]
            + "\n& & \multicolumn{2}{c}{\\textbf{Test Accuracy (\%)}} &  \\\\"
            + latex[top_rule_index:]
        )
    else:
        print("❌ Could not find the row to insert the midrule.")

    # Find \label{tab} and vspace after it
    search_term = r"\label{tab:tuned-benchmark}"
    index_to_insert_vspace = latex.find(search_term) + len(search_term)
    if index_to_insert_vspace != -1:
        latex = (
            latex[:index_to_insert_vspace]
            + "\n\\vspace{3.5pt}"
            + latex[index_to_insert_vspace:]
        )
    else:
        print("❌ Could not find the row to insert the vspace.")

    return latex


def aggregate(df, param_tuples, metric="mean"):
    df_agg = df.groupby(param_tuples).agg({("eval", "test/acc"): [metric]})
    return df_agg


def plot_heatmap_on_ax(
    ax,
    results,
    param1_values,
    param2_values,
    cmap,
    xlabels=False,
    ylabels=False,
    vmin=0,
    vmax=1,
    pos="NW",
):
    cax = ax.matshow(results, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_yticks(
        np.arange(len(param1_values)),
        param1_values if ylabels else [],
        fontsize=22
    )
    ax.set_xticks(
        np.arange(len(param2_values)),
        param2_values if xlabels else [],
        rotation="vertical",
        fontsize=22,
    )

    ax.tick_params(
        axis="both",
        right= "E" in pos,
        left="W" in pos ,
        bottom="S" in pos,
        top="N" in pos,
        labelbottom= "S" in pos,
        labeltop= "N" in pos,
        labelleft= "W" in pos,
        labelright= "E" in pos,
    )
    ax.grid(False)
    return cax


def calcualte_vs(df_runs, params, metric="mean", vmin=None, vmax=None):
    if vmin != None and vmax != None:
        return vmin, vmax

    values = []
    for param1 in params:
        for param2 in params:
            if param1 == param2:
                continue
            df_agg = aggregate(df_runs, [param1, param2], metric)[
                ("eval", "test/acc", metric)
            ].unstack()
            values.extend(df_agg.values.flatten())

    vmin = min(values) if vmin is None else vmin
    vmax = max(values) if vmax is None else vmax
    return vmin, vmax


# grid of (n-1 x n-1) plots, n is number of hyperparameters, each plot is a heatmap of mean test acc for different hyperparameter value combinations
def grid(
    df_runs,
    params,
    metric="mean",
    cmap="YlGn",
    vmin=None,
    vmax=None,
    figsize=(10, 10),
    rename_dict=None,
):
    n = len(params)
    fig, axs = plt.subplots(nrows=n - 1, ncols=n - 1, figsize=figsize)
    fig.tight_layout(pad=3.0)

    rename = lambda x: rename_dict[x] if rename_dict and x in rename_dict else x

    vmin, vmax = calcualte_vs(df_runs, params, metric, vmin=vmin, vmax=vmax)
    for i in range(n - 1):
        for j in range(1, i + 1):
            axs[i][j - 1].axis("off")
        for j in range(i + 1, n):
            param1 = params[i]
            param2 = params[j]
            ax = axs[i, j - 1]
            df_agg = aggregate(df_runs, [param1, param2], metric)[
                ("eval", "test/acc", metric)
            ].unstack()
            cax = plot_heatmap_on_ax(
                axs[i][j - 1],
                df_agg,
                df_agg.index,
                df_agg.columns,
                cmap,
                i == 0,
                j == n - 1,
                vmin=vmin,
                vmax=vmax,
                pos="NE",
            )

            # print labels
            if i == j - 1:
                # position lable on top
                axs[i][j - 1].set_xlabel(
                    rename(param2[1]), fontsize=22, fontweight="bold"
                )
                # position the label a bit down
                if j != n - 1:
                    axs[i][j - 1].xaxis.set_label_coords(0.5, -0.3)
                if i == 0:
                    axs[i][j - 1].set_ylabel(
                        rename(param1[1]), fontsize=22, fontweight="bold", rotation=0
                    )

                    axs[i][j - 1].yaxis.set_label_coords(-0.2, 0.4)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    for ax in axs.flat:
        ax.set_anchor("NE")

    # colorbar anchor to the left
    ticks = np.linspace(vmin, vmax, 5)
    cb = fig.colorbar(
        cax, ax=axs[-1, 0:2], orientation="horizontal", aspect=15, ticks=ticks,
    )
    # # legend with red dot and best hyperparameters
    # cb.ax.legend([plt.scatter([], [], marker='o', color='red')], ['best hyperparameters'], loc='lower center', ncol=2,
    #              fontsize=16, frameon=False, bbox_to_anchor=(0.5, 1))
    # colorbar ticks
    cb.ax.set_xticklabels([f"{tick:.2f}" for tick in ticks], fontsize=22)
    return fig


def combined_grid(
    df1,
    df2,
    params,
    metric="mean",
    cmap1="YlGn",
    cmap2="YlGn",
    vmin=None,
    vmax=None,
    figsize=(10, 10),
    rename_dict=None,
):
    n = len(params)
    fig, axs = plt.subplots(nrows=n, ncols=n, figsize=figsize)
    fig.tight_layout(pad=3.0)

    rename = lambda x: rename_dict[x] if rename_dict and x in rename_dict else x

    vmin1, vmax1 = calcualte_vs(df1, params, metric, vmin=vmin, vmax=vmax)
    vmin2, vmax2 = calcualte_vs(df2, params, metric, vmin=vmin, vmax=vmax)

    for i in range(n):
        # top part
        for j in range(i + 1, n):
            param1 = params[i]
            param2 = params[j]
            ax = axs[i][j]
            df_agg = aggregate(df1, [param1, param2], metric)[
                ("eval", "test/acc", metric)
            ].unstack()
            cax1 = plot_heatmap_on_ax(
                ax,
                df_agg,
                [rename(x) for x in df_agg.index],
                [rename(x) for x in df_agg.columns],
                cmap1,
                i == 0,
                j == 0,
                vmin=vmin1,
                vmax=vmax1,
            )

        # bottom part
        for j in range(0, i):
            ax = axs[i][j]
            param1 = params[i]
            param2 = params[j]
            df_agg = aggregate(df2, [param1, param2], metric)[
                ("eval", "test/acc", metric)
            ].unstack()
            cax2 = plot_heatmap_on_ax(
                ax,
                df_agg,
                [rename(x) for x in df_agg.index],
                [rename(x) for x in df_agg.columns],
                cmap2,
                i == 0,
                j == 0,
                vmin=vmin2,
                vmax=vmax2,
                pos="SW",
            )

        # labels
        ax = axs[i][i]
        ax.set_facecolor("white")
        param = params[i]
        df_agg1 = aggregate(df1, [param, param], metric)[
            ("eval", "test/acc", metric)
        ].unstack()
        df_agg2 = aggregate(df2, [param, param], metric)[
            ("eval", "test/acc", metric)
        ].unstack()
        param1_values = df_agg1.index
        param2_values = df_agg2.columns
        # mat  of nans of (df1 param i value count) x (df2 param i value count)
        mat = np.full((len(param1_values), len(param2_values)), np.nan)
        plot_heatmap_on_ax(
            ax,
            mat,
            [rename(x) for x in param1_values],
            [rename(x) for x in param2_values],
            cmap1,
            i==0,
            i==0,
            pos="NW",
            )
        ax.set_xlabel(
            rename(param[1]),
            fontsize=22,
            fontweight="bold",
            verticalalignment="center",
            horizontalalignment="center",
        )
        ax.xaxis.set_label_coords(0.5, 0.5)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    for ax in axs.flat:
        ax.set_anchor("NE")

    ticks1 = np.linspace(vmin1, vmax1, 5)
    #vertical on the right
    newax1 = fig.add_axes([0.75, 0, 0.3, 1], anchor='N')
    newax1.axis("off")
    cb1 = fig.colorbar(
        cax1, ax=newax1, orientation="vertical", aspect=10, ticks=ticks1,
    )
    cb1.ax.set_yticklabels([f"{tick:.0f}" for tick in ticks1], fontsize=22)

    #horizontal on the bottom
    ticks2=np.linspace(vmin2, vmax2, 5)
    newax2 = fig.add_axes([0, -0.03, 1, 0.3], anchor='N')
    newax2.axis("off")
    cb2 = fig.colorbar(
        cax2, ax=newax2, orientation="horizontal", aspect=10, ticks=ticks2
    )
    cb2.ax.set_xticklabels([f"{tick:.0f}" for tick in ticks2], fontsize=22)

    return fig


def visualise_hparams(df, dataset=None, method=None, hparam="lr", ax=None):
    if not ax:
        _, ax = plt.subplots(figsize=(5, 5))
    only_sot = df[("config", "use_sot")]
    is_dataset = df[("config", "dataset")] == dataset if dataset else True
    is_method = df[("config", "method")] == method if method else True
    filtered_df = df[only_sot & is_dataset & is_method]

    sns.barplot(
        filtered_df,
        x=("hparams", hparam),
        y=("eval", "test/acc"),
        ax=ax,
    )
    ax.set_xlabel(hparam)
    ax.set_ylabel("Test Acc. (%)")
