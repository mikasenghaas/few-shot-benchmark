import os
import time
from abc import abstractmethod

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_and_extract_archive


# Inspired/modified from WILDS (https://wilds.stanford.edu/)
# and COMET (https://github.com/snap-stanford/comet)


class FewShotDataset(Dataset):
    """
    Base class for all datasets (both simple classifier and few-shot) in this project.
    It supports downloading and extracting the dataset if it does not exist locally and performs
    some basic checks (class attributes are set, data directory exists, etc.).

    Specifies that the following methods and properties must be implemented by subclasses:
    - __getitem__
    - __len__
    - dim
    - get_data_loader
    """

    def __init__(self):
        """
        Initializes the FewShotDataset by running some basic checks.
        """
        self.check_init()

    def check_init(self):
        """
        Convenience function to check that the FewShotDataset is properly configured.
        """
        required_attrs = ["_dataset_name", "_data_dir"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise ValueError(f"FewShotDataset must have attribute {attr}.")

        if not os.path.exists(self._data_dir):
            raise ValueError(
                f"{self._data_dir} does not exist yet. Please generate/download the dataset first."
            )

    @abstractmethod
    def __getitem__(self, i):
        """Returns a sample from the dataset."""
        return NotImplemented

    @abstractmethod
    def __len__(self):
        """Returns the length of the dataset."""
        return NotImplemented

    @property
    @abstractmethod
    def dim(self):
        """Returns the number of features/ dimensionality of the dataset."""
        return NotImplemented

    @abstractmethod
    def get_data_loader(self, mode="train") -> DataLoader:
        """
        Returns a DataLoader for the dataset.
        """
        return NotImplemented

    @property
    def dataset_name(self):
        """
        A string that identifies the dataset, e.g., 'tabula_muris'
        """
        return self._dataset_name

    @property
    def data_dir(self):
        """
        A string that specifies the path to the dataset directory.
        """
        return self._data_dir

    def set_n_episodes(self):
        """
        Sets the number of episodes for the dataset to match the number of samples
        in the dataset.
        """
        assert (
            hasattr(self, "num_samples")
            and hasattr(self, "n_way")
            and hasattr(self, "n_support")
        ), "Please set the number of samples, number of classes and number of support samples per class first."
        self.n_episodes = self.num_samples // (self.n_way * self.n_support)

    def initialize_data_dir(self, root_dir: str, download_flag: bool = True):
        """
        Utility function to initialize the data directory. Checks
        if the dataset directory (root_dir/dataset_name) exists and
        if not, downloads the dataset to the directory if the
        download flag is set to True.

        Args:
            root_dir (str): path to the directory where to save the data
            download_flag (bool): whether to download the dataset if it does not exist locally

        Returns:
            None
        """
        os.makedirs(root_dir, exist_ok=True)
        self._data_dir = os.path.join(root_dir, self._dataset_name)
        if not self._dataset_exists_locally():
            if not download_flag:
                raise FileNotFoundError(
                    f"The {self._dataset_name} dataset could not be found in {self._data_dir}. Please"
                    f" download manually. "
                )

            self._download_dataset()

    def _download_dataset(self):
        """
        Downloads and extracts the dataset to the data directory.
        """
        if self._dataset_url is None:
            raise ValueError(
                f"{self._dataset_name} cannot be automatically downloaded. Please download it manually."
            )

        print(f"Downloading dataset to {self._data_dir}...")

        try:
            start_time = time.time()
            download_and_extract_archive(
                url=self._dataset_url,
                download_root=self._data_dir,
                remove_finished=True,
            )
            download_time_in_minutes = (time.time() - start_time) / 60
            print(
                f"\nIt took {round(download_time_in_minutes, 2)} minutes to download and uncompress the dataset.\n"
            )
        except Exception as e:
            print("Exception: ", e)

    def _dataset_exists_locally(self):
        """
        Checks if the dataset exists locally.
        """
        return os.path.exists(self._data_dir) and (
            len(os.listdir(self._data_dir)) > 0 or (self._dataset_url is None)
        )


class FewShotSubDataset(Dataset):
    """
    Simple PyTorch Dataset that wraps a subset of samples that all belong to the same class.
    """

    def __init__(self, samples: np.ndarray, category: int):
        """
        Initializes the FewShotSubDataset with the given samples and category.

        Args:
            samples (np.ndarray): samples that belong to the same class
            category (int): category of the samples
        """
        self.samples = samples
        self.category = category

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        """
        Returns a single sample and its category.

        Args:
            i (int): index of the sample to return

        Returns:
            sample (np.ndarray): sample
            category (int): category of the sample
        """
        return self.samples[i], self.category

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            len (int): number of samples in the dataset
        """
        return self.samples.shape[0]

    @property
    def dim(self):
        """
        Returns the number of features

        Returns:
            dim (int): dimensionality of the data
        """
        return self.samples.shape[1]


class EpisodicBatchSampler(object):
    """
    Simple iterator/ sampler that generates batches of episodes.
    In each episode it samples n_way random classes from the class
    indices {0, ..., n_classes-1}. The total length of the iterator
    is n_episodes.
    """

    def __init__(self, n_classes, n_way, n_episodes):
        """
        Initialises the EpisodicBatchSampler with the given parameters.

        Args:
            n_classes (int): number of classes in the dataset
            n_way (int): number of classes in each episode
            n_episodes (int): number of episodes
        """
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        """
        Returns the number of episodes.

        Returns:
            n_episodes (int): number of episodes
        """
        return self.n_episodes

    def __iter__(self):
        """
        Returns the episodic class indices for each episode.

        Returns:
            class_indices (torch.Tensor): class indices per episode
        """
        for _ in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[: self.n_way]
