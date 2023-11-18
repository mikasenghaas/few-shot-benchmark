import os
from abc import ABC

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.cell.utils import MacaData, MacaDataImproved
from datasets.dataset import FewShotDataset, FewShotSubDataset


class TMDataset(FewShotDataset, ABC):
    """
    Abstract base class for the Tabula Muris dataset.

    Sets the class attributes (dataset name and URL) and implements a utility
    method to load the entire dataset.
    """

    _dataset_name = "tabula_muris"
    _dataset_url = "http://snap.stanford.edu/comet/data/tabula-muris-comet.zip"

    def load_tabular_muris(self, mode="train", min_samples=20):
        """
        Loads the Tabula Muris dataset from the data directory. Depending on the
        mode (train, val, test), we only return the samples and targets in the
        specified tissues (e.g. BAT for train, Skin for val, etc.)

        The min_samples parameter filters out all classes with less than the
        specified number. This is necessary to not include tissues with less
        samples than needed for a k-shot learning task.

        Args:
            mode (str): train, val, or test
            min_samples (int): minimum number of samples per class

        Returns:
            samples (np.ndarray): samples in the dataset
            targets (np.ndarray): labels for the samples
        """
        train_tissues = [
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
        val_tissues = ["Skin", "Lung", "Thymus", "Aorta"]
        test_tissues = ["Large_Intestine", "Marrow", "Pancreas", "Tongue"]
        split = {"train": train_tissues, "val": val_tissues, "test": test_tissues}
        adata = MacaData(
            src_file=os.path.join(self._data_dir, "tabula-muris-comet.h5ad")
        ).adata
        tissues = split[mode]
        # subset data based on target tissues
        adata = adata[adata.obs["tissue"].isin(tissues)]

        filtered_index = (
            adata.obs.groupby(["label"])
            .filter(lambda group: len(group) >= min_samples)
            .reset_index()["index"]
        )
        adata = adata[filtered_index]

        # convert gene to torch tensor x
        samples = adata.to_df().to_numpy(dtype=np.float32)
        # convert label to torch tensor y
        targets = adata.obs["label"].cat.codes.to_numpy(dtype=np.int32)
        # go2gene = get_go2gene(adata=adata, GO_min_genes=32, GO_max_genes=None, GO_min_level=6, GO_max_level=1)
        # go_mask = create_go_mask(adata, go2gene)
        return samples, targets


class TMDatasetImproved(FewShotDataset, ABC):
    """
    An improved version of the TMDataset base class that uses
    the MacaDataImproved class to load the data. The class only loads and
    pre-processes the data within the split (train, val, test) which significantly
    improves the loading time. Additionally, it introduces the subset parameter
    which allows to only 10% of the data for faster prototyping.

    Sets the class attributes (dataset name and URL) and implements a utility
    method to load the entire dataset.
    """

    _dataset_name = "tabula_muris"
    _dataset_url = "http://snap.stanford.edu/comet/data/tabula-muris-comet.zip"

    def load_tabular_muris(self, mode="train", min_samples=20, subset=False):
        """
        Loads the Tabula Muris dataset from the data directory. Depending on the
        mode (train, val, test), we only return the samples and targets in the
        specified tissues (e.g. BAT for train, Skin for val, etc.)

        The min_samples parameter filters out all classes with less than the
        specified number. This is necessary to not include tissues with less
        samples than needed for a k-shot learning task.

        Allows to only load a subset of the data (10%) for faster prototyping.

        Args:
            mode (str): train, val, or test
            min_samples (int): minimum number of samples per class
            subset (bool): whether to subset the data

        Returns:
            samples (np.ndarray): samples in the dataset
            targets (np.ndarray): labels for the samples
        """

        # Load all the data in the split
        path = os.path.join(self._data_dir, "tabula-muris-comet.h5ad")
        adata = MacaDataImproved(src_file=path, mode=mode, subset=subset).adata

        # Filter out classes with less than min_samples (typically set to k-shot)
        filtered_index = (
            adata.obs.groupby(["label"])
            .filter(lambda group: len(group) >= min_samples)
            .reset_index()["index"]
        )
        adata = adata[filtered_index]

        # Convert features and targets to numpy arrays
        samples = adata.X
        targets = adata.obs["label"].cat.codes.to_numpy(dtype=np.int32)

        return samples, targets


class TMSimpleDataset(TMDatasetImproved):
    def __init__(self, batch_size, root="./data/", mode="train", min_samples=20):
        self.initialize_data_dir(root, download_flag=True)
        self.samples, self.targets = self.load_tabular_muris(mode, min_samples)
        self.batch_size = batch_size
        super().__init__()

    def __getitem__(self, i):
        return self.samples[i], self.targets[i]

    def __len__(self):
        return self.samples.shape[0]

    @property
    def dim(self):
        return self.samples.shape[1]

    def get_data_loader(self) -> DataLoader:
        data_loader_params = dict(
            batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)

        return data_loader


class TMSetDataset(TMDatasetImproved):
    def __init__(
        self, n_way, n_support, n_query, n_episode=100, root="./data", mode="train"
    ):
        self.initialize_data_dir(root, download_flag=True)

        self.n_way = n_way
        self.n_episode = n_episode
        min_samples = n_support + n_query

        samples_all, targets_all = self.load_tabular_muris(mode, min_samples)
        self.categories = np.unique(targets_all)  # Unique cell labels
        self.x_dim = samples_all.shape[1]

        self.sub_dataloader = []

        sub_data_loader_params = dict(
            batch_size=min_samples,
            shuffle=True,
            num_workers=0,  # use main thread only or may receive multiple batches
            pin_memory=False,
        )
        for cl in self.categories:
            samples = samples_all[targets_all == cl, ...]
            sub_dataset = FewShotSubDataset(samples, cl)
            self.sub_dataloader.append(
                torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params)
            )

        super().__init__()

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.categories)

    @property
    def dim(self):
        return self.x_dim

    def get_data_loader(self) -> DataLoader:
        sampler = EpisodicBatchSampler(len(self), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)
        return data_loader
