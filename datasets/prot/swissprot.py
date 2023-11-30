import numpy as np
import pickle
import os
from abc import ABC
from dataclasses import dataclass

from dataclasses_json import dataclass_json
from torch.utils.data import Dataset, DataLoader

import torch
from datasets.dataset import FewShotDataset, EpisodicBatchSampler
from datasets.prot.utils import (
    get_samples_using_ic,
    get_samples,
    check_min_samples,
    get_mode_ids,
    get_ids,
    encodings,
)

EMB_PATH = "embeddings"
EMB_LAYER = 33
PROTDIM = 1280


@dataclass_json
@dataclass
class ProtSample:
    input_seq: torch.Tensor
    annot: torch.Tensor
    entry: str


class SPDataset(FewShotDataset, ABC):
    """
    Abstract base class for the SwissProt dataset.

    Sets the class attributes (dataset name and URL) and implements a utility
    method to load the entire dataset.
    """

    _dataset_name = "swissprot"
    _dataset_url = "https://drive.google.com/u/0/uc?id=1a3IFmUMUXBH8trx_VWKZEGteRiotOkZS&export=download"

    def load_swissprot(
        self,
        level: int = 5,  # Not used
        mode: str = "train",
        min_samples: int = 20,
        subset: bool = 1.0,
        seed: int = 42,
    ):
        """
        Loads the SwissProt dataset from the data directory. If the data has
        already been processed, it will be loaded from the processed directory.
        Otherwise, the raw data will be processed and saved in the processed
        directory.

        The min_samples parameter filters out all classes with less than the
        specified number. This is necessary to not include tissues with less
        samples than needed for a k-shot learning task.

        Args:
            level (int): level of the SwissProt hierarchy
            mode (str): train, val, or test
            min_samples (int): minimum number of samples per class
            subset (float): ratio of the data to load (e.g. 0.1 for 10%)

        Returns:
            samples (np.ndarray): list of SwissProt samples (including id, input sequence and target)
        """
        # Load all samples from the data directory
        processed_path = os.path.join(self.data_dir, "processed", "swissprot.pkl")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        if not os.path.exists(processed_path):
            print("Did not find processed data. Processing SwissProt data now...")
            samples = get_samples_using_ic(root=self.data_dir)
            pickle.dump(samples, open(processed_path, "wb"))

        # Load the processed samples
        samples = pickle.load(open(processed_path, "rb"))

        # Filter out samples that are not in the split (train, val, test)
        unique_ids = set(get_mode_ids(samples)[mode])
        samples = [sample for sample in samples if sample.annot in unique_ids]

        # Subset the data
        if subset != 1.0:
            np.random.seed(seed)
            subset_size = int(len(samples) * subset)
            random_indices = np.random.choice(
                len(samples), size=subset_size, replace=False
            )
            samples = list(np.array(samples)[random_indices])

        # Filter out classes with less than min_samples
        samples = check_min_samples(samples, min_samples)

        return samples


class SPSimpleDataset(SPDataset):
    """
    Simple SwissProt dataset that loads the entire (processed) dataset into memory and wraps
    inside a PyTorch Dataset object. Supports functionality for retrieving a single sample,
    a batched data loader and the dimensionality of the data.
    """

    def __init__(
        self,
        batch_size: int,
        root: str = "./data/",
        mode: str = "train",
        min_samples: int = 20,
        subset: float = 1.0,
    ):
        """
        Initializes the dataset by loading the entire dataset into memory and all encodings
        into memory. Note, that the data is not downloaded automatically and must be downloaded
        manually from the link in the `_dataset_url` class attribute.

        Args:
            batch_size (int): batch size for the data loader
            root (str): path to the data directory to download the raw data in. (Default: `./data/`)
            mode (str): train, val, or test
            subset (float): ratio of the data to load (e.g. 0.1 for 10%)
            min_samples (int): minimum number of samples per class
        """
        # Initialise the data directory
        self.initialize_data_dir(root, download_flag=False)  # Download manually

        # Loads the data and target encoding mapping
        self.samples = self.load_swissprot(
            mode=mode,
            min_samples=min_samples,
            subset=subset,
        )
        self.trg2idx = encodings(self.data_dir)

        # Save parameters
        self.batch_size = batch_size

        # Call the parent class constructor (FewShotDataset)
        super().__init__()

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """
        Returns a single sample and its target

        Args:
            i (int): index of the sample to return

        Returns:
            sample (torch.Tensor): sample
            target (int): target
        """
        sample = self.samples[i]
        return sample.input_seq, self.trg2idx[sample.annot]

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset

        Returns:
            len (int): number of samples in the dataset
        """
        return len(self.samples)

    @property
    def dim(self):
        """
        Returns the dimensionality in the dataset (protein sequence length)

        Returns:
            dim (int): dimensionality of the data
        """
        return self.samples[0].input_seq.shape[0]

    def get_data_loader(
        self, shuffle: bool = True, num_workers: int = 4, pin_memory: bool = True
    ) -> DataLoader:
        """
        Returns a PyTorch DataLoader object for the dataset.

        Args:
            shuffle (bool): whether to shuffle the data
            num_workers (int): number of workers to use for loading the data
            pin_memory (bool): whether to pin the memory

        Returns:
            data_loader (DataLoader): PyTorch DataLoader object for the dataset
        """
        data_loader_params = dict(
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        data_loader = DataLoader(self, **data_loader_params)

        return data_loader


class SPSetDataset(SPDataset):
    """
    Few-shot SwissProt dataset that loads the entire (processed) dataset into memory and wraps
    inside a PyTorch Dataset object. Supports functionality for retrieving episodes (batches) and
    the dimensionality of the data.
    """

    def __init__(
        self,
        n_way: int,
        n_support: int,
        n_query: int,
        n_episodes: int | None = None,
        root: str = "./data",
        mode: str = "train",
        subset: float = 1.0,
    ):
        """
        Initializes the dataset by loading the entire dataset into memory.
        If the data is not present in the data directory, it is downloaded to the `root` directory, processed and
        loaded first. Also creates a list of sub-datasets, one for each class in the dataset with
        a corresponding data loader using the `FewShotSubDataset` class. The data loader always
        samples n_support + n_query samples in a single batch. The number of episodes per epoch
        is set automatically such that the number of samples seen per epoch is equal to the total
        number of samples in the dataset.

        Args:
            n_way (int): number of classes in a single episode
            n_support (int): number of support samples per class (k-shot)
            n_query (int): number of query samples per class
            n_episodes (optional, int): number of episodes per epoch (Default: n_samples / (n_way * n_support)
            root (str): path to the data directory to download the raw data in. (Default: `./data/`)
            mode (str): train, val, or test
            subset (float): ratio of the data to load (e.g. 0.1 for 10%)
        """
        # Initialise the data directory
        self.initialize_data_dir(root, download_flag=False)

        # Save all constructor arguments as attributes
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.min_samples = n_support + n_query

        # Save encoding
        self.trg2idx = encodings(self.data_dir)

        # Load all samples
        samples = self.load_swissprot(
            mode=mode,
            min_samples=self.min_samples,
            subset=subset,
        )
        self.num_samples = len(samples)
        self.annotations = get_ids(samples)

        # Set the number of episodes
        if n_episodes:
            self.n_episodes = n_episodes
        else:
            self.set_n_episodes()

        # Create a list of sub-datasets, one for each class in the dataset
        sub_data_loader_params = dict(
            batch_size=self.min_samples,
            shuffle=True,
            num_workers=0,  # use main thread only or may receive multiple batches
            pin_memory=False,
        )

        # Iterate over all unique classes to create a sub-dataset and data loader for each class
        self.sub_dataloader = []
        for annot in self.annotations:
            sub_samples = [sample for sample in samples if sample.annot == annot]
            sub_dataset = SubDataset(sub_samples, self.trg2idx)
            sub_dataloader = DataLoader(sub_dataset, **sub_data_loader_params)
            self.sub_dataloader.append(sub_dataloader)

        # Call the parent class constructor (FewShotDataset)
        super().__init__()

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a batch of samples and targets for a single episode (few-shot task) for
        a single class. The batch size is always n_support + n_query.

        Args:
            i (int): index of the class to return

        Returns:
            samples (torch.Tensor): batch of samples
            targets (torch.Tensor): batch of targets
        """
        return next(iter(self.sub_dataloader[i]))

    def __len__(self) -> int:
        """
        Returns the number of classes in the dataset.

        Returns:
            len (int): number of classes in the dataset
        """
        return len(self.annotations)

    @property
    def dim(self):
        """
        Returns the number of features (encoded sequence length) in the dataset

        Returns:
            dim (int): dimensionality of the data
        """
        return PROTDIM

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
        sampler = EpisodicBatchSampler(len(self), self.n_way, self.n_episodes)
        data_loader_params = dict(
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)
        return data_loader


class SubDataset(Dataset):
    """
    Dataset for a single class in the SwissProt dataset. Wraps a list of samples and
    returns a single sample and target when indexed.
    """

    def __init__(self, samples, encoder):
        self.samples = samples
        self.encoder = encoder

    def __getitem__(self, i):
        sample = self.samples[i]
        return sample.input_seq, self.encoder[sample.annot]

    def __len__(self):
        return len(self.samples)

    @property
    def dim(self):
        return PROTDIM


if __name__ == "__main__":
    d = SPSetDataset(5, 5, 15)
