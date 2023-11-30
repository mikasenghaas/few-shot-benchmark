import numpy as np
import pandas as pd
import scanpy as sc
from anndata import read_h5ad


class MacaData:
    """
    Utility class to load and preprocess Tabula Muris dataset.
    """

    def __init__(self, path: str):
        """
        Loads and preprocesses the Tabula Muris dataset from the the src_file
        path using `anndata` library and preprocesses the data using `scanpy`.

        Args:
            path (str): path to the .h5ad file containing the dataset

        Returns:
            None
        """
        # Load original data
        self.adata = read_h5ad(path).copy()

        # Define target (cell type)
        self.target = "cell_ontology_class_reannotated"

        # Compute ground truth mapping and add encoded targets to the dataset
        targets = list(self.adata.obs[self.target])
        unique_targets = sorted(set(targets))

        self.trg2idx = {trg: idx for idx, trg in enumerate(unique_targets)}
        self.idx2trg = {idx: trg for idx, trg in enumerate(unique_targets)}

        # Add ground truth labels to the dataset
        targets = list(self.adata.obs[self.target])
        target_idxs = [self.trg2idx[target] for target in targets]
        self.adata.obs["label"] = pd.Categorical(values=target_idxs)

        # Preprocess the data
        self.process_data()

    def process_data(self):
        """
        Processes the data using scanpy. It performs the following steps:
            - Filter out cells with no target
            - Filter out genes that are expressed in less than 5 cells
            - Filter out cells with less than 5000 counts and 500 genes expressed
            - Normalize per cell (simple lib size normalization)
            - Filter out genes with low dispersion (retain the once with high variance)
            - Log transform and scale the data
            - Zero-imputation of Nans

        Args:
            None

        Returns:
            None
        """
        adata = self.adata.copy()

        # Filter out cells with no target
        adata.obs[self.target] = adata.obs[self.target].astype(str)
        missing_target = adata.obs[self.target].isin(["nan", "NA"])
        adata = adata[~missing_target, :].copy()

        # Filter out genes with less than min_cells having non-zero expression
        sc.pp.filter_genes(adata, min_cells=5)

        # Filter out cells with less than 5000 counts and 500 genes expressed
        sc.pp.filter_cells(adata, min_counts=5000)
        sc.pp.filter_cells(adata, min_genes=500)

        # Normalize per cell (lib size normalization)
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)

        # Filter out genes with low dispersion
        adata.raw = adata
        adata = sc.pp.filter_genes_dispersion(
            adata,
            subset=False,
            min_disp=0.5,
            max_disp=None,
            min_mean=0.0125,
            max_mean=10,
            n_bins=20,
            n_top_genes=None,
            log=True,
            copy=True,
        )
        adata = adata[:, adata.var.highly_variable].copy()

        # Log-transform the data
        sc.pp.log1p(adata)

        # Scale the data
        sc.pp.scale(adata, max_value=10, zero_center=True)

        # Zero-imputation of Nans
        adata.X[np.isnan(adata.X)] = 0

        self.processed = True
        self.adata = adata
