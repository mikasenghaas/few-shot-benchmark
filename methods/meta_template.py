from abc import abstractmethod
from abc import ABC
from typing import Iterable, Union, List, Tuple
from tqdm import tqdm
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

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
        self.SOT = sot

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

    def parse_feature(
        self, x: Union[torch.Tensor, List[torch.Tensor]], is_feature: bool
    ) -> torch.Tensor:
        """
        [Helper] Return split into support and query sets.
        Important: if parse feature is False, we first run the input tensor through the backbone.

        Args:
            x (torch.Tensor): input tensor
            is_feature (bool): whether the input tensor is a feature tensor or not

        Returns:
            z_support (torch.Tensor): support set
            z_query (torch.Tensor): query set
        """

        # Make sure that each input tensor has gradient computation enabled
        if isinstance(x, list):
            x = [Variable(obj.to(self.device)) for obj in x]
        else:
            x = Variable(x.to(self.device))

        # Get a 3D tensor of shape (n_way, n_support + n_query, feat_dim)
        if is_feature:
            z_all = x
        else:
            # First flatten the second dimension of the input tensor
            # See the docstring of reshape2feature for more details
            if isinstance(x, list):
                x = [self.reshape2feature(x) for obj in x]
            else:
                x = self.reshape2feature(x)

            # Extract features using the backbone
            z_all = self.feature.forward(x)

            # Apply SOT if provided
            if self.SOT is not None:
                z_all = self.SOT(z_all)  # Returns square matrix

            # Now reshape back the tensor to (n_way, n_support + n_query, feat_dim)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)

        # Split the tensor into support and query sets
        z_support = z_all[:, : self.n_support]
        z_query = z_all[:, self.n_support :]

        return z_support, z_query

    def correct(self, x: torch.Tensor) -> Tuple[float, int]:
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
        scores = self.set_forward(x)

        # Get the labels of the query set
        # Ex. n_way = 2, n_query = 3 -> y_query = [0, 0, 0, 1, 1, 1]
        y_query = self.get_episode_labels(self.n_query, enable_grad=False).cpu().numpy()

        # Argmax the scores to get the predictions, i.e., for each
        # query sample, we get the class with the highest score
        _, topk_labels = scores.data.topk(k=1, dim=1, largest=True, sorted=True)
        topk_ind = topk_labels.cpu().numpy()

        # Compute the number of correct predictions
        top1_correct = np.sum(topk_ind[:, 0] == y_query)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        [Backbone] Extract features using the backbone of the model.

        Args:
            x (torch.Tensor): input 2D tensor

        Returns:
            out (torch.Tensor): tensor
        """

        out = self.feature.forward(x)
        return out

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
        for i, (x, _) in pbar:
            # Set the number of query samples and classes
            self.set_nquery(x)
            if self.change_way:
                self.set_nway(x)

            # Run one iteration of the optimization process
            optimizer.zero_grad()
            batch_loss = self.set_forward_loss(x)
            batch_loss.backward()
            optimizer.step()

            # Add batch loss to total loss
            loss += batch_loss.item()

            # Print the loss
            self.log_training_progress(pbar, epoch, i, num_episodes, loss)

        # Compute the epoch loss as average of the episode losses
        epoch_loss = loss / num_episodes

        return epoch_loss

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
        for i, (x, _) in pbar:
            # Set the number of query samples and classes
            self.set_nquery(x)
            if self.change_way:
                self.set_nway(x)

            # Compute the accuracy
            correct, preds = self.correct(x)
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

    def adapt(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        support: Tuple[torch.Tensor, torch.Tensor],
        query: torch.Tensor,
        loss_function: torch.nn.Module,
        batch_size: int = 4,
    ):
        """
        [Finetune] Finetune the model by freezing the backbone and training a new softmax clasifier.
        The query set is used to evaluate the model.

        Args:
            model (torch.nn.Module): model to finetune
            optimizer (torch.optim.Optimizer): optimizer
            support (Tuple[torch.Tensor, torch.Tensor]): support set
            query (torch.Tensor): query set
            loss_function (torch.nn.Module): loss function
            batch_size (int): batch size

        Returns:
            scores (torch.Tensor): scores of the query set of shape (n_way * n_query, n_way)
        """

        # Get the support features and labels
        z_support, y_support = support

        # Get the size of the support set
        support_size = self.n_way * self.n_support
        assert z_support.size(0) == support_size, "Error: support set size is incorrect"

        # Finetune the classifier
        for _ in range(100):
            # Shuffle the support set
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                # Reset the gradients
                optimizer.zero_grad()

                # Select the batch from the support set
                selected_id = torch.from_numpy(
                    rand_id[i : min(i + batch_size, support_size)]
                ).to(self.device)

                # Get the embeddings and labels for the batch
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]

                # Compute the predictions and loss
                scores = model(z_batch)
                loss = loss_function(scores, y_batch)

                # Backpropagate the loss
                loss.backward()
                optimizer.step()

        # Compute the final predictions for the query set
        scores = model(query)

        return scores

    def set_forward_adaptation(self, x, is_feature=True):
        """
        [Finetuning] Further of the model by freezing the backbone and training a new softmax clasifier
        on the provided input's support set. The query set is used to evaluate the model.

        Args:
            x (torch.Tensor): input tensor
            is_feature (bool): whether the input tensor is a feature tensor or not

        Returns:
            scores (torch.Tensor): scores of the query set of shape (n_way * n_query, n_way)
        """

        # Split the input into support and query sets (is_feature = True --> do not run backbone)
        assert is_feature == True, "Feature is fixed in further adaptation"
        z_support, z_query = self.parse_feature(x, is_feature)

        # Flatten the second dimension of the query and support sets
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # Create the labels of the support set
        y_support = self.get_episode_labels(self.n_support, enable_grad=True)

        # Set up the new softmax classifier
        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.to(self.device)

        # Set up the optimizer
        set_optimizer = torch.optim.SGD(
            linear_clf.parameters(),
            lr=0.01,
            momentum=0.9,
            dampening=0.9,
            weight_decay=0.001,
        )

        # Set up the loss function
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(self.device)

        # Finetune the classifier
        scores = self.adapt(
            linear_clf,
            set_optimizer,
            (z_support, y_support),
            z_query,
            loss_function,
            batch_size=4,
        )

        return scores
