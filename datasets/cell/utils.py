import numpy as np
import pandas as pd
import scanpy as sc
from anndata import read_h5ad


class MacaData:
    """
    Utility class to load and preprocess the Tabula Muris dataset.
    """

    def __init__(
        self,
        src_file,
        filter_genes=True,
    ):
        """
        Loads the Tabula Muris dataset from the data directory specified in src_file
        using the `anndata` library and preprocesses the data using `scanpy`.

        Args:
            src_file (str): path to the .h5ad file containing the dataset
            filter_genes (bool): whether to filter out genes with low expression

        Returns:
            None
        """
        # Loads entire dataset
        self.adata = read_h5ad(src_file).copy()
        self.annotation_type = "cell_ontology_class_reannotated"

        # Convert annotation column to string

        # Create copy to avoid implicit modification warnings
        self.adata.obs[self.annotation_type] = self.adata.obs[
            self.annotation_type
        ].astype(str)

        # Filter out cells with no annotation
        self.adata = self.adata[
            ~self.adata.obs[self.annotation_type].isin(["nan", "NA"]), :
        ].copy()

        # Add ground truth labels
        self.cells2names = self.cellannotation2ID(self.annotation_type)

        # Filter out genes with less than min_cells having non-zero expression
        if filter_genes:
            sc.pp.filter_genes(self.adata, min_cells=5)

        # Preprocess data (filter cells, normalize, log, scale)
        self.adata = self.preprocess_data(self.adata)

    def preprocess_data(self, adata):
        """
        Preprocesses the data using scanpy. It performs the following steps:
            - Filter out cells with less than 5000 counts and 500 genes expressed
            - Normalize per cell (simple lib size normalization)
            - Filter out genes with low dispersion (retain the once with high variance)
            - Log transform and scale the data
            - Zero-imputation of Nans

        Args:
            adata (anndata.AnnData): the dataset to preprocess

        Returns:
            adata (anndata.AnnData): the preprocessed dataset
        """
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

        return adata

    def get_tissue_data(self, tissue, age=None):
        """
        Select data for given tissue.
        filtered: if annotated return only cells with annotations, if unannotated return only cells without labels, else all
        age: '3m','18m', '24m', if None all ages are included
        """

        tiss = self.adata[self.adata.obs["tissue"] == tissue, :]

        if age:
            return tiss[tiss.obs["age"] == age]

        return tiss

    def cellannotation2ID(self, annotation_type):
        """Adds ground truth clusters data."""
        annotations = list(self.adata.obs[annotation_type])
        annotations_set = sorted(set(annotations))

        mapping = {a: idx for idx, a in enumerate(annotations_set)}

        truth_labels = [mapping[a] for a in annotations]
        self.adata.obs["label"] = pd.Categorical(values=truth_labels)
        # 18m-unannotated
        #
        return mapping


class MacaDataImproved(MacaData):
    """
    Improved MacaData class that allows to load only tissue data for a
    given split and subset the data to 10% of the original size. Inherits
    from the MacaData class to use the same pre-processing and annotation
    mappings but overrides the constructor.
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

    def __init__(
        self,
        annotation_type="cell_ontology_class_reannotated",
        src_file="dataset/cell_data/tabula-muris-senis-facs-official-annotations.h5ad",
        filter_genes=True,
        subset=False,
        mode="train",
    ):
        """
        Loads the Tabula Muris dataset from the data directory specified in src_file
        using the `anndata` library and preprocesses the data using `scanpy`.

        Args:
            src_file (str): path to the .h5ad file containing the dataset
            filter_genes (bool): whether to filter out genes with low expression

        Returns:
            None
        """
        # Load all data and change the type of the annotation column to string
        self.adata = read_h5ad(src_file)
        self.adata.obs[annotation_type] = self.adata.obs[annotation_type].astype(str)

        # Load annotations to ID mapping
        self.cells2names = self.cellannotation2ID(annotation_type)

        # Filter our cells with no annotation
        self.adata = self.adata[self.adata.obs[annotation_type] != "nan", :]
        self.adata = self.adata[self.adata.obs[annotation_type] != "NA", :]

        # Subset data to only include cells with annotation
        mode2tissues = {
            "train": self._train_tissues,
            "val": self._val_tissues,
            "test": self._test_tissues,
        }
        self.adata = self.adata[
            self.adata.obs["tissue"].isin(mode2tissues[mode])
        ].copy()

        # Subset the loaded data
        if subset:
            subset_size = len(self.adata) // 10
            random_indices = np.random.choice(
                self.adata.shape[0], size=subset_size, replace=False
            )
            self.adata = self.adata[random_indices, :].copy()

        # Filter out genes with less than min_cells having non-zero expression
        if filter_genes:
            sc.pp.filter_genes(self.adata, min_cells=5)

        # Preprocess data (filter cells, normalize, log, scal)
        self.adata = self.preprocess_data(self.adata)
