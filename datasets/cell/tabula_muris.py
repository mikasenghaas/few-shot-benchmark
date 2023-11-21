import pickle
import os
from abc import ABC

import numpy as np
from torch.utils.data import DataLoader

from datasets.cell.utils import MacaData
from datasets.dataset import FewShotDataset, FewShotSubDataset, EpisodicBatchSampler

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

# Subset data to only include cells with annotation
mode2tissues = {
    "train": train_tissues,
    "val": val_tissues,
    "test": test_tissues,
}


class TMDataset(FewShotDataset, ABC):
    """
    Abstract base class for the Tabula Muris dataset.

    Sets the class attributes (dataset name and URL) and implements a utility
    method to load the entire dataset.

    Uses the MacaDataLoader class to load the processed data directly for the
    split and implments allows to only load a subset of the data (10%) for faster
    experimentation.
    """

    _dataset_name = "tabula_muris"
    _dataset_url = "http://snap.stanford.edu/comet/data/tabula-muris-comet.zip"

    def load_tabular_muris(
        self, mode: str = "train", min_samples: int = 20, subset: float = 1.0
    ):
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
            subset (float): ratio of the data to load (e.g. 0.1 for 10%)

        Returns:
            samples (np.ndarray): samples in the dataset
            targets (np.ndarray): labels for the samples
        """

        # Process the data if it is not present in the data directory
        processed_data_path = os.path.join(
            self.data_dir, "processed", "tabula-muris.pkl"
        )
        mapping_path = os.path.join(self.data_dir, "processed", "mapping.pkl")
        processed_dir = os.path.dirname(processed_data_path)

        os.makedirs(processed_dir, exist_ok=True)
        if not os.path.exists(processed_data_path) or not os.path.exists(mapping_path):
            print("Did not find processed data. Processing TabulaMuris data now...")
            # Load and pre-process the data
            path = os.path.join(self._data_dir, "tabula-muris-comet.h5ad")
            maca_data = MacaData(path)

            # Save the processed data and mappings
            pickle.dump(maca_data.adata, open(processed_data_path, "wb"))
            pickle.dump(maca_data.trg2idx, open(mapping_path, "wb"))

        # Load the processed data and encodings
        self.data = pickle.load(open(processed_data_path, "rb"))
        # mapping = pickle.load(open(mapping_path, "rb"))

        # Filter out samples from the specified tissues
        tissues = mode2tissues[mode]
        tissue_filter = self.data.obs["tissue"].isin(tissues)
        self.data = self.data[tissue_filter]

        # Filter out classes with less than min_samples (typically set to k-shot)
        filtered_index = (
            self.data.obs.groupby(["label"])
            .filter(lambda group: len(group) >= min_samples)
            .reset_index()["index"]
        )
        self.data = self.data[filtered_index]

        # Subset the data
        subset_size = int(len(self.data) * subset)
        random_indices = np.random.choice(
            self.data.shape[0], size=subset_size, replace=False
        )
        self.data = self.data[random_indices, :].copy()

        # Convert features and targets to numpy arrays
        samples = self.data.X
        targets = self.data.obs["label"].cat.codes.to_numpy(dtype=np.int32)

        return samples, targets


class TMSimpleDataset(TMDataset):
    """
    Simple Tabula Muris dataset that loads the entire (processed) dataset into memory and wraps
    inside a PyTorch Dataset object. Supports functionality for retrieving a single sample,
    a batched data loader and the dimensionality of the data.
    """

    def __init__(
        self,
        batch_size: int,
        root: str = "./data/",
        mode: str = "train",
        subset: float = 1.0,
        min_samples: int = 20,
    ):
        """
        Initializes the dataset by loading the entire dataset into memory. If the data is not
        present in the data directory, it is downloaded to the `root` directory, processed and
        loaded first.

        Args:
            batch_size (int): batch size for the data loader
            root (str): path to the data directory to download the raw data in. (Default: `./data/`)
            mode (str): train, val, or test
            subset (float): ratio of the data to load (e.g. 0.1 for 10%)
            min_samples (int): minimum number of samples per class
        """
        self.initialize_data_dir(root, download_flag=True)
        self.samples, self.targets = self.load_tabular_muris(mode, min_samples, subset)
        self.batch_size = batch_size
        super().__init__()

    def __getitem__(self, i) -> tuple[np.ndarray, int]:
        """
        Returns a single sample (gene expression) and its target (cell type label)

        Args:
            i (int): index of the sample to return

        Returns:
            sample (np.ndarray): gene expression of the sample
            target (int): cell type label of the sample
        """
        return self.samples[i], self.targets[i]

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset

        Returns:
            len (int): number of samples in the dataset
        """
        return self.samples.shape[0]

    @property
    def dim(self) -> int:
        """
        Returns the number of features (genes) in the dataset

        Returns:
            dim (int): dimensionality of the data
        """
        return self.samples.shape[1]

    # TODO: A cleaner API would probably be to also get batch_size as function argument, instead of in the constructor
    def get_data_loader(
        self, shuffle: bool = True, num_workers: int = 4, pin_memory: bool = True
    ) -> DataLoader:
        """
        Returns a PyTorch DataLoader object that can be used to iterate over the dataset in batches
        of size `batch_size` (specified in constructor). The DataLoader is shuffled by default and
        uses 4 workers to load the data in parallel. For GPU, pin_memory should be set to True. For CPU,
        set pin_memory to False and disable multi-processing by setting num_workers to 0.

        Args:


        Returns:
            data_loader (DataLoader): PyTorch DataLoader object
        """
        data_loader_params = dict(
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,  # For GPU: num_workers=4, pin_memory=True
        )
        data_loader = DataLoader(self, **data_loader_params)

        return data_loader


class TMSetDataset(TMDataset):
    """
    Few-shot Tabula Muris dataset that loads the entire (processed) dataset into memory and wraps
    inside a PyTorch Dataset object. Supports functionality for retrieving episodes (batches) and
    the dimensionality of the data.
    """

    def __init__(
        self,
        n_way: int,
        n_support: int,
        n_query: int,
        n_episode: int = 100,
        root: str = "./data",
        mode: str = "train",
        subset: float = 1.0,
    ):
        """
        Initializes the dataset by loading the entire dataset into memory. If the data is not
        present in the data directory, it is downloaded to the `root` directory, processed and
        loaded first. Also creates a list of sub-datasets, one for each class in the dataset with
        a corresponding data loader using the `FewShotSubDataset` class. The data loader always
        samples n_support + n_query samples in a single batch.
        """
        # Save parameters
        self.n_way = n_way
        self.n_episode = n_episode
        min_samples = n_support + n_query  # Need at least this many samples per class

        # Download the data if it is not present in the data directory
        self.initialize_data_dir(root, download_flag=True)

        # Load the data
        samples, targets = self.load_tabular_muris(mode, min_samples, subset=subset)

        # Get the unique cell labels
        self.unique_targets = np.unique(targets)

        # Initialise empty list of data loader for each class
        self.sub_dataloader = []
        sub_data_loader_params = dict(
            batch_size=min_samples,
            shuffle=True,
            num_workers=0,  # use main thread only or may receive multiple batches
            pin_memory=False,
        )

        # Iterate over all unique classes to create a sub-dataset and data loader for each class
        for target in self.unique_targets:
            sub_samples = samples[targets == target, ...]
            sub_dataset = FewShotSubDataset(sub_samples, target)
            sub_dataloader = DataLoader(sub_dataset, **sub_data_loader_params)
            self.sub_dataloader.append(sub_dataloader)

        # Call parent constructor which does some checks
        super().__init__()

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a batch of samples and targets for a single episode (few-shot task) for
        a single class. The batch size is always n_support + n_query.

        Args:
            i (int): index of the class to return

        Returns:
            samples (np.ndarray): batch of samples for the episode
            targets (np.ndarray): batch of targets for the episode
        """
        return next(iter(self.sub_dataloader[i]))

    def __len__(self) -> int:
        """
        Returns the number of classes in the dataset.

        Returns:
            len (int): number of classes in the dataset
        """
        return len(self.unique_targets)

    @property
    def dim(self) -> int:
        """
        Returns the number of features (genes) in the dataset

        Returns:
            dim (int): dimensionality of the data
        """
        return self.samples_all.shape[1]

    def get_data_loader(
        self, num_workers: int = 4, pin_memory: bool = True
    ) -> DataLoader:
        """
        Returns an episodic data loader that can be used to iterate over the dataset in episodes.

        Args:
            num_workers (int): number of workers to load the data in parallel
            pin_memory (bool): whether to pin the data to GPU memory

        Returns:
            data_loader (DataLoader): PyTorch DataLoader object
        """
        sampler = EpisodicBatchSampler(len(self), self.n_way, self.n_episode)
        data_loader_params = dict(
            batch_sampler=sampler, num_workers=num_workers, pin_memory=pin_memory
        )
        data_loader = DataLoader(self, **data_loader_params)
        return data_loader
