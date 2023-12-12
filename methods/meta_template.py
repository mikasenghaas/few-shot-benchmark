from abc import abstractmethod
from abc import ABC
from typing import Iterable, Union, List, Tuple
from tqdm import tqdm
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder

from .sot import SOT


class MetaTemplate(nn.Module, ABC):
    def __init__(
        self,
        backbone: torch.nn.Module,
        n_way: int,
        n_support: int,
        change_way: bool = True,
        print_freq: int = 1,
        type: str = "classification",
        sot: SOT = None,
    ):
        """
        Base class for the meta-learning methods.

        Args:
            backbone: feature extractor
            n_way: number of classes in a task
            n_support: number of support samples per class
            change_way: whether the number of classes would change in a task, e.g. 5-way -> 20-way
            print_freq (int): how often (in terms of # of batches) to print the results
            type (str): the type of the task (classification or regression)
            sot (SOT) : Self-Optimal Transport Feature Transformer
            save_intermediates (bool): Whether to save intermediate results during set_forward()
        """

        # Init parent directory
        super(MetaTemplate, self).__init__()

        # Init class variables
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.feature = backbone
        self.feat_dim = (
            self.feature.final_feat_dim if sot is None else sot.final_feat_dim
        )
        self.change_way = change_way  # some methods allow different_way classification during training and test
        self.type = type
        self.sot = sot

        # Init device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define logging
        self.print_freq = print_freq

    @abstractmethod
    def set_forward(self, x, is_feature=False):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def reshape2feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        [Helper] Reshape a XD tensor to a (X - 1)D tensor by flattening the second dimension.

        Example 1: we are given 3D tensor of shape (5, 10, 20). We want to reshape it to a 2D tensor of shape (50, 20).
        The input dimensions in this case would be (n_way, n_support + n_query, feat_dim).

        Example 2: we are given 4D tensor of shape (5, 10, 20, 30). We want to reshape it to a 3D tensor of shape (50, 20, 30).
        The input dimensions in this case would be (n_way, n_support + n_query, channel, feat_dim).

        Args:
            x (torch.Tensor): input tensor

        Returns:
            x (torch.Tensor): reshaped tensor
        """

        x = x.contiguous().view(
            self.n_way * (self.n_support + self.n_query), *x.size()[2:]
        )

        return x

    def reshape2set(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape a XD tensor to a (X + 1)D tensor by splitting the first dimension into two dimensions.
        The first dimension is split into n_way and n_support + n_query.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            x (torch.Tensor): reshaped tensor
        """
        return x.contiguous().view(self.n_way, self.n_support + self.n_query, -1)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone. Requires the input tensor to be 2D (batch_size, feat_dim).
        For set-based methods, make sure to first flatten the second dimension of the input tensor
        using reshape2feature() which will return a tensor of shape (n_way * n_support + n_query, feat_dim).

        Args:
            x (torch.Tensor): input tensor

        Returns:
            out (torch.Tensor): output tensor
        """
        assert x.ndim == 2, "Input tensor must be 2D. Call reshape2feature() first."
        return self.feature.forward(x)

    def forward_sot(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone. Requires the input tensor to be 2D (batch_size, feat_dim).
        For set-based methods, make sure to first flatten the second dimension of the input tensor
        using reshape2feature() which will return a tensor of shape (n_way * n_support + n_query, feat_dim).

        Args:
            x (torch.Tensor): input tensor

        Returns:
            out (torch.Tensor): output tensor
        """
        assert x.ndim == 2, "Input tensor must be 2D. Call reshape2feature() first."
        assert self.sot is not None, "SOT is not enabled."

        return self.sot(x)

    def parse_feature(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Return split into support and query sets.

        Args:
            x (torch.Tensor): input tensor
            is_feature (bool): whether the input tensor is a feature tensor or not

        Returns:
            z_support (torch.Tensor): support set
            z_query (torch.Tensor): query set
        """
        assert x.ndim == 3, "Input tensor must be 3D. Call reshape2set() first."

        # Split the tensor into support and query sets
        z_support = x[:, : self.n_support].reshape(self.n_way * self.n_support, -1)
        z_query = x[:, self.n_support :].reshape(self.n_way * self.n_query, -1)

        return z_support, z_query

    def correct(self, x: torch.Tensor, y) -> Tuple[float, int]:
        """
        [Helper] Compute number of correct predictions.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            top1_correct (float): number of correct predictions
            len(y_query) (int): number of predictions (n_way * n_query)
        """

        # Run inference of the model on the input
        # Scores is of shape (n_way * n_query, n_way)
        outputs = self.set_forward(x)
        scores = outputs["scores"]
        preds = scores.argmax(dim=1)

        # Get the labels of the query set
        # Ex. n_way = 2, n_query = 3 -> y_query = [0, 0, 0, 1, 1, 1]
        # y_query = self.get_episode_labels(self.n_query, enable_grad=False).cpu().numpy()
        y_query = y[:, self.n_support :].reshape(self.n_way * self.n_query)

        # Argmax the scores to get the predictions, i.e., for each
        # query sample, we get the class with the highest score

        # Compute the number of correct predictions
        top1_correct = (preds == y_query).sum()

        return float(top1_correct), len(y_query)

    def set_nquery(self, x: Union[torch.Tensor, List[torch.Tensor]]):
        """
        [Helper] Set the number of query samples based on the input and
        the number of support samples

        Args:
            x (torch.Tensor): input tensor
        """

        if isinstance(x, list):
            self.n_query = x[0].size(1) - self.n_support
        else:
            self.n_query = x.size(1) - self.n_support

    def set_nway(self, x: Union[torch.Tensor, List[torch.Tensor]]):
        """
        [Helper] Set the number of classes based on the input

        Args:
            x (torch.Tensor): input tensor
        """

        if isinstance(x, list):
            self.n_way = x[0].size(0)
        else:
            self.n_way = x.size(0)

    def get_progress_bar(
        self,
        iterable: Iterable,
        total: Union[int, None] = None,
    ) -> tqdm:
        terminal_width = shutil.get_terminal_size().columns
        description_width = max(terminal_width - 30 - 20, 0)

        return tqdm(
            iterable,
            total=total,
            bar_format="{desc:"
            + str(description_width)
            + "}{percentage:3.0f}%|{bar:"
            + str(30)
            + "}| {n_fmt}/{total_fmt}",
        )

    def log_training_progress(
        self, pbar, epoch: int, i: int, n: int, loss: float, few_shot: bool = True
    ):
        """
        [Helper] Log the training progress.

        Args:
            pbar (tqdm): tqdm object
            epoch (int): current epoch
            i (int): current batch / episode
            n (int): total number of batches / episodes
            loss (float): accumulated loss
            few_shot (bool): whether the training is few-shot or not
        """

        if (i + 1) % self.print_freq == 0:
            current_loss = loss / float(i + 1)
            description = (
                "Training: Epoch {:03d} | {} {:04d}/{:04d} | Loss {:.5f}".format(
                    epoch + 1,
                    "Episode" if few_shot else "Batch",
                    i + 1,
                    n,
                    current_loss,
                )
            )
            pbar.set_description(description)

    def eval_test_performance(
        self, evals: Union[List[Tuple[float, int]], List[float]]
    ) -> Tuple[float, float]:
        """
        [Helper] Log the evaluation performance.

        For classification we use accuracy and for regression we use pearson correlation.

        Args:
            evals (List[Tuple[float, int]]):
                classification: list of tuples (number of correct predictions, number of predictions)
                regression: list of tuples (predictions, ground truth)

        Returns:
            metric_mean (float): the mean of the metric (accuracy, correlation)
            metric_std (float): the standard deviation of the metric (accuracy, correlation)
        """

        # Number of episodes that are evaluated
        n_episodes = len(evals)

        # Compute the mean and standard deviation of the accuracies
        acc_all = np.asarray([x[0] / x[1] * 100 for x in evals])
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        acc_ci = 1.96 * (acc_std / np.sqrt(n_episodes))

        print("Few-shot Accuracy = %4.2f%% +- %4.2f%%" % (acc_mean, acc_ci))

        return acc_mean, acc_ci, acc_std

    def get_episode_labels(self, n: int, enable_grad: bool = True) -> torch.Tensor:
        """
        [Helper] Return the labels of for the support / query set.

        Args:
            n (int): number of samples in the set
            enable_grad (bool): whether to enable gradient computation or not
        Returns:
            y_support (torch.Tensor): support set labels

        Example:
            n_way = 2, n_support = 3 -> y_support = [0, 0, 0, 1, 1, 1]
        """

        # Create the labels of the support set
        y = torch.from_numpy(np.repeat(range(self.n_way), n))

        # ensure they have gradient computation enabled
        if enable_grad:
            y = Variable(y.to(self.device))

        return y

    def train_loop(
        self,
        epoch: int,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
    ):
        """
        [MetaLearning] Train the model for one epoch.

        Args:
            epoch (int): current epoch
            train_loader (torch.utils.data.DataLoader): training data loader
            optimizer (torch.optim.Optimizer): optimizer
        """

        # Run one epoch of episodic training
        num_episodes = len(train_loader)
        pbar = self.get_progress_bar(enumerate(train_loader), total=num_episodes)
        pbar.set_description(
            f"Training: Epoch {epoch:03d} | Episodes 000/{num_episodes:03d} | 0.0000"
        )
        loss = 0.0
        for i, (x, y) in pbar:
            # Set the number of query samples and classes
            self.set_nquery(x)
            if self.change_way:
                self.set_nway(x)

            # Reshuffle
            x, y = self.shuffle_queries(x, y)

            # Integer encode
            mapper = {c.item(): i for i, c in enumerate(y[:, 0])}
            y = y.apply_(lambda x: mapper[x])

            # Run one iteration of the optimization process
            optimizer.zero_grad()
            batch_loss = self.set_forward_loss(x, y)
            batch_loss.backward()
            optimizer.step()

            # Add batch loss to total loss
            loss += batch_loss.item()

            # Print the loss
            self.log_training_progress(pbar, epoch, i, num_episodes, loss)

        # Compute the epoch loss as average of the episode losses
        epoch_loss = loss / num_episodes

        return epoch_loss

    def shuffle_queries(self, x, y):
        """
        [MetaLearning] Shuffle the query set.

        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): label tensor

        Returns:
            x (torch.Tensor): shuffled input tensor
            y (torch.Tensor): shuffled label tensor
        """
        x_support, x_query = self.parse_feature(x)
        y_support, y_query = (
            y[:, : self.n_support].flatten(),
            y[:, self.n_support :].flatten(),
        )

        rand_id = np.random.permutation(self.n_way * self.n_query)
        x_query = x_query[rand_id]
        y_query = y_query[rand_id]

        x_support = x_support.reshape(self.n_way, self.n_support, -1)
        y_support = y_support.reshape(self.n_way, self.n_support)
        x_query = x_query.reshape(self.n_way, self.n_query, -1)
        y_query = y_query.reshape(self.n_way, self.n_query)

        x = torch.cat((x_support, x_query), dim=1)
        y = torch.cat((y_support, y_query), dim=1)

        return x, y

    def test_loop(self, test_loader: torch.utils.data.DataLoader) -> (float, float):
        """
        [MetaLearning Eval] Evaluate the model on the test set via
        few-shot accuracy.

        Args:
            test_loader (DataLoader): the test data loader

        Returns:
            acc_mean (float): the mean accuracy over all episodes
            acc_ci (float): the 95% confidence interval of the accuracy
            acc_std (float): the standard deviation of the accuracy
        """

        # Collect the accuracy for each episode
        evals = []
        num_batches = len(test_loader)
        pbar = self.get_progress_bar(enumerate(test_loader), total=num_batches)
        pbar.set_description(f"Testing: Episodes 000/{num_batches:03d} | 0.0000")
        total_correct, total_preds = 0, 0
        for i, (x, y) in pbar:
            # Set the number of query samples and classes
            self.set_nquery(x)
            if self.change_way:
                self.set_nway(x)

            # Reshuffle
            x, y = self.shuffle_queries(x, y)

            # Integer encode
            mapper = {c.item(): i for i, c in enumerate(y[:, 0])}
            y = y.apply_(lambda x: mapper[x])

            # Compute the accuracy
            correct, preds = self.correct(x, y)
            evals.append([correct, preds])

            total_correct += correct
            total_preds += preds

            pbar.set_description(
                f"Evaluating: Episodes {i+1:03d}/{num_batches:03d} | Running Acc. {(100 * total_correct / total_preds):.2f}%"
            )

        # Compute the mean and standard deviation of the metric
        acc_mean, acc_ci, acc_std = self.eval_test_performance(evals)

        # Return the mean and (standard deviation) of the accuracy
        return acc_mean, acc_ci, acc_std
