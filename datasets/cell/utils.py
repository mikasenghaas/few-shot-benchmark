import os
import json
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

        # Set processed flag to False
        self.processed = False

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

    def save_processed_data(self, dir: str):
        """
        Saves the processed data to the specified path. Raises an error if the
        data has not been processed yet.

        Args:
            dir (str): path to the directory where to save the data

        Returns:
            None
        """
        os.makedirs(dir, exist_ok=True)
        dst_file = os.path.join(dir, "tabula-muris-comet.h5ad")
        dst_mapping = os.path.join(dir, "trg2idx.json")
        if self.processed:
            self.adata.write_h5ad(dst_file)
            with open(dst_mapping, "w") as f:
                json.dump(self.trg2idx, f)
        else:
            raise ValueError("Data has not been processed yet.")


class MacaDataLoader:
    """
    Wrapper around the MacaData class that allows to load only tissue data for a
    specific mode (train, val, test) and subset the data to a fraction of the original size.
    """

    _train_tissues = [
        "BAT",
        "Bladder",
        "Brain_Myeloid",
        "Brain_Non-Myeloid",
        "Diaphragm",
        "GAT",
        "Heart",
        "Kidney",
        "Limb_Muscle",
        "Liver",
        "MAT",
        "Mammary_Gland",
        "SCAT",
        "Spleen",
        "Trachea",
    ]
    _val_tissues = ["Skin", "Lung", "Thymus", "Aorta"]
    _test_tissues = ["Large_Intestine", "Marrow", "Pancreas", "Tongue"]

    # Subset data to only include cells with annotation
    _mode2tissues = {
        "train": _train_tissues,
        "val": _val_tissues,
        "test": _test_tissues,
    }

    def __init__(
        self,
        path: str,
        mode: str = "train",
        subset: bool = False,
        seed: int = 42,
    ):
        """
        Loads the Tabula Muris dataset from the data directory specified in src_file
        using the `anndata` library and preprocesses the data using `scanpy`.

        Args:
            path (str): path to the original .h5ad data
            annotation_type (str): the type of annotation to use as ground truth
            mode (str): train, val, or test
            subset (bool): whether to subset the data to 10% of the original size
            seed (int): seed for the random number generator

        Returns:
            None
        """
        # Get relevant paths to processed data
        path_dir = os.path.join(os.path.dirname(path), "processed")
        processed_path = os.path.join(path_dir, "tabula-muris-comet.h5ad")
        processed_mappings = os.path.join(path_dir, "trg2idx.json")

        # Load the processed data if it exists, otherwise process the raw data and save it
        if not os.path.exists(processed_path) or not os.path.exists(processed_mappings):
            print("Processed data not found. Processing raw data...")
            maca_data = MacaData(path)
            maca_data.preprocess_data()
            maca_data.save_processed_data(path_dir)

        # Save parameters
        self.mode = mode
        self.subset = subset
        np.random.seed(seed)

        # Load the processed data and mappings
        self.adata = self._load_data(processed_path)
        self.trg2idx, self.idx2trg = self._load_mappings(processed_mappings)

        # Filter out cells of tissue that are not in the specified mode
        self.adata = self._filter_tissue(mode)

        # Subset the loaded data
        if subset:
            self.adata = self._subset_data()

    def _load_data(self, path):
        return read_h5ad(path).copy()

    def _load_mappings(self, path):
        with open(path, "r") as f:
            trg2idx = json.load(f)

        idx2trg = {idx: trg for trg, idx in trg2idx.items()}

        return trg2idx, idx2trg

    def _filter_tissue(self, mode):
        """Filters out cells from the specified tissue."""
        tissues = self._mode2tissues[mode]
        tissue_filter = self.adata.obs["tissue"].isin(tissues)

        return self.adata[tissue_filter, :]

    def _subset_data(self):
        """Subset the data to 10% of the original size."""
        subset_size = len(self.adata) // 10
        random_indices = np.random.choice(
            self.adata.shape[0], size=subset_size, replace=False
        )
        return self.adata[random_indices, :].copy()
