from abc import abstractmethod
from abc import ABC
from typing import Union, List, Tuple
from IPython.display import clear_output
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable

from utils.data_utils import pearson_corr


class MetaTemplate(nn.Module, ABC):
    def __init__(
            self, 
            backbone : torch.nn.Module, 
            n_way : int, 
            n_support : int, 
            change_way : bool = True,
            log_wandb : bool = True,
            print_freq : int = 10,
            type : str = "classification"
        ):
        """
        Base class for the meta-learning methods.

        Args:
            backbone: feature extractor
            n_way: number of classes in a task
            n_support: number of support samples per class
            change_way: whether the number of classes would change in a task, e.g. 5-way -> 20-way
            log_wandb (bool): whether to log the results to wandb
            print_freq (int): how often (in terms of # of batches) to print the results
            type (str): the type of the task (classification or regression)
        """

        # Init parent directory
        super(MetaTemplate, self).__init__()

        # Init class variables
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.feature = backbone
        self.feat_dim = self.feature.final_feat_dim
        self.change_way = change_way  # some methods allow different_way classification during training and test
        self.type = type

        # Init device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define logging
        self.log_wandb = log_wandb
        self.print_freq = print_freq

    @abstractmethod
    def set_forward(self, x, is_feature=False):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def reshape2feature(self, x : torch.Tensor) -> torch.Tensor:
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

    def parse_feature(self, x : Union[torch.Tensor, List[torch.Tensor]], is_feature : bool) -> torch.Tensor:
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

            # Now reshape back the tensor to (n_way, n_support + n_query, feat_dim)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        
        # Split the tensor into support and query sets
        z_support = z_all[:, : self.n_support]
        z_query = z_all[:, self.n_support :]

        return z_support, z_query

    def correct(self, x : torch.Tensor) -> Tuple[float, int]:
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
        y_query = np.repeat(range(self.n_way), self.n_query)

        # Argmax the scores to get the predictions, i.e., for each
        # query sample, we get the class with the highest score
        _, topk_labels = scores.data.topk(k=1, dim=1, largest=True, sorted=True)
        topk_ind = topk_labels.cpu().numpy()

        # Compute the number of correct predictions
        top1_correct = np.sum(topk_ind[:, 0] == y_query)

        return float(top1_correct), len(y_query)
    
    def set_nquery(self, x : Union[torch.Tensor, List[torch.Tensor]]):
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
    
    def set_nway(self, x : Union[torch.Tensor, List[torch.Tensor]]):
        """
        [Helper] Set the number of classes based on the input

        Args:
            x (torch.Tensor): input tensor
        """

        if isinstance(x, list):
            self.n_way = x[0].size(0)
        else:
            self.n_way = x.size(0)
    
    def log_training_progress(self, epoch : int, i : int, n : int, avg_loss : float):
        """
        [Helper] Log the training progress.

        Args:
            epoch (int): current epoch
            i (int): current batch
            n (int): total number of batches
            avg_loss (float): average loss
        """

        if i % self.print_freq == 0:
            clear_output(wait=True)
            current_loss = avg_loss / float(i + 1)
            print(
                "ℹ️ epoch {:d} | batch {:d}/{:d} | loss {:f}".format(
                    epoch, i, n, current_loss
                )
            )
            if self.log_wandb: wandb.log({"loss/train": current_loss})
    
    def eval_test_performance(self, evals : Union[List[Tuple[float, int]], List[float]]) -> Tuple[float, float]:
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

        if self.type == "classification":
            # Compute the size of test set
            n = sum([x[1] for x in evals])

            # Compute the mean and standard deviation of the accuracies
            acc_all = np.asarray([x[0] / x[1] * 100 for x in evals])
            acc_mean = np.mean(acc_all)
            acc_std = np.std(acc_all)

            print(
                "%d Test Acc = %4.2f%% +- %4.2f%%"
                % (n, acc_mean, 1.96 * acc_std / np.sqrt(n))
            )

            return acc_mean, acc_std

        else:
            # Compute the mean and standard deviation of the correlations
            corr_mean = np.mean(evals)
            corr_std = np.std(evals)
            n = len(evals)

            print(
                "%d Test Corr = %4.2f%% +- %4.2f%%"
                % (n, corr_mean, 1.96 * corr_std / np.sqrt(n))
            )

            return corr_mean, corr_std

    def correlation(self, x : torch.Tensor, y : torch.Tensor, type : str = "pearson") -> float:
        """
        [Helper] Compute the correlation between the predictions and the ground truth.

        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): ground truth tensor
            type (str): type of correlation to compute (default: pearson)
        
        Returns:
            corr (float): correlation
        """

        # Run inference of the model on the input    
        y_pred = self.set_forward(x, y).reshape(-1).to(self.device)

        # Get the values of the query set
        y_query = y[:, self.n_support :].reshape(-1).to(self.device)

        # Compute the correlation
        if type == "pearson":
            corr = pearson_corr(y_pred, y_query)
        else:
            raise ValueError(f"Correlation type {type} not defined")

        return corr.cpu().detach().numpy()

    def get_support_labels(self, enable_grad : bool = True) -> torch.Tensor:
        """
        [Helper] Return the labels of the support set.

        Example: n_way = 2, n_support = 3 -> y_support = [0, 0, 0, 1, 1, 1]

        Returns:
            y_support (torch.Tensor): support set labels
        """

        # Create the labels of the support set
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))

        # ensure they have gradient computation enabled
        if enable_grad:
            y_support = Variable(y_support.to(self.device))

        return y_support

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        [Backbone] Extract features using the backbone of the model.

        Args:
            x (torch.Tensor): input 2D tensor
        
        Returns:
            out (torch.Tensor): tensor
        """

        out = self.feature.forward(x)
        return out

    def train_loop(self, epoch : int, train_loader : torch.utils.data.DataLoader, optimizer : torch.optim.Optimizer):
        """
        [MetaLearning] Train the model for one epoch.

        Args:
            epoch (int): current epoch
            train_loader (torch.utils.data.DataLoader): training data loader
            optimizer (torch.optim.Optimizer): optimizer 
        """

        # Run one epoch of episodic training
        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):

            # Set the number of query samples and classes
            self.set_nquery(x)
            if self.change_way: self.set_nway(x)

            # Run one iteration of the optimization process 
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()

            # Log the loss
            avg_loss += loss.item()

            # Print the loss
            self.log_training_progress(epoch, i, len(train_loader), avg_loss)

    def test_loop(self, test_loader : torch.utils.data.DataLoader, return_std : bool = False) -> (float, float):
        """
        [MetaLearning Eval] Evaluate the model on the test set. We use
        accuracy for classification and pearson correlation for regression.

        Args:
            test_loader (DataLoader): the test data loader
            return_std (bool): whether to return the standard deviation of the target metric.
        
        Returns:
            metric_mean (float): the mean of the metric (accuracy, correlation)
            metric_std (float): the standard deviation of the metric (accuracy, correlation)
        """

        # Collect the accuracy for each episode
        evals = []
        for data in tqdm(test_loader, desc="Batches evaluated:"):

            # Parse the input according to the task type 
            if self.type == "classification":
                x, _ = data
            else:
                x, y = data

            # Set the number of query samples and classes
            self.set_nquery(x)
            if self.change_way: self.set_nway(x)

            # Compute the accuracy
            if self.type == "classification":
                total_correct, total_preds = self.correct(x)
                evals.append([total_correct, total_preds])
            
            # Compute the pearson correlation
            else:
                raise NotImplementedError("Regression not implemented yet")
        
        # Compute the mean and standard deviation of the metric
        metric_mean, metric_std = self.eval_test_performance(evals)

        # Return the mean and (standard deviation) of the accuracy
        if return_std:
            return metric_mean, metric_std
        else:
            return metric_mean


    def adapt(self, model : torch.nn.Module, optimizer : torch.optim.Optimizer, support : Tuple[torch.Tensor, torch.Tensor], query : torch.Tensor, loss_function : torch.nn.Module, batch_size : int = 4):
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
        y_support = self.get_support_labels()

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
            linear_clf, set_optimizer, (z_support, y_support), z_query, loss_function, batch_size=4
        )

        return scores
