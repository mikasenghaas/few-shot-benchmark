# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml
from typing import List, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable

from backbones.blocks import Linear_fw
from methods.meta_template import MetaTemplate


class MAML(MetaTemplate):
    def __init__(
        self,
        backbone: torch.nn.Module,
        n_way: int,
        n_support: int,
        n_task: int,
        task_update_num: int,
        inner_lr: float,
        **kwargs,
    ):
        """
        MAML implementation. TODO: add more explanation

        Args:
            backbone (torch.nn.Module) : backbone network for the method
            n_way (int) : number of classes in a task
            n_support (int) : number of support samples per class
            n_task (int) : number of tasks to train on
            task_update_num (int) : number of gradient updates to perform on each task
            inner_lr (float) : learning rate for the inner loop
        """

        # Init the parent class
        super(MAML, self).__init__(
            backbone, n_way, n_support, change_way=False, **kwargs
        )

        # Define the classifier which comes after the backbone
        self.classifier = Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)

        # Define the loss function
        if n_way == 1:
            raise ValueError("MAML does not support regression tasks")
        else:
            self.type = "classification"
            self.loss_fn = nn.CrossEntropyLoss()

        # Define after how many tasks to update the "slow" parameters
        self.n_task = n_task

        # Define how many gradient updates to perform "fast" weights (adaptation)
        self.task_update_num = task_update_num

        # Define the inner (adaptation) learning rate
        self.inner_lr = inner_lr

        # Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def parse_feature(
        self, x: Union[List[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the input into support and query sets and flatten.

        Args:
            x (Union[List[torch.Tensor], torch.Tensor]) : input of shape (n_way, n_support + n_query, feat_dim)

        Returns:
            x_support (torch.Tensor) : support set of shape (n_way * n_support, feat_dim)
            x_query (torch.Tensor) : query set of shape (n_way * n_query, feat_dim)
            y_support (torch.Tensor) : support set labels of shape (n_way * n_support,)
        """
        # Run backbone and possibly SOT
        # (shape: (n_way, n_support, feat_dim)), ...
        x_support, x_query = super().parse_feature(x, is_feature=False)

        # Flatten
        x_support = x_support.contiguous().view(self.n_way * self.n_support, -1)
        x_query = x_query.contiguous().view(self.n_way * self.n_query, -1)

        # Enable gradients
        x_support = x_support.requires_grad_()
        x_query = x_query.requires_grad_()

        # Get the labels of the support set
        y_support = self.get_episode_labels(self.n_support, enable_grad=True)

        return x_support, x_query, y_support

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Run backbone and classifier on input data.

        Args:
            x (Union[List[torch.Tensor], torch.Tensor]) : input data of shape (batch_size, *)

        Returns:
            scores (torch.Tensor) : scores of shape (n_way * n_query, n_way)
        """
        scores = self.classifier.forward(x)
        return scores

    def set_forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Run backbone and classifier on input data and perform adaptation.

        Args:
            x (Union[List[torch.Tensor], torch.Tensor]) : input data of shape (batch_size, *)

        Returns:
            scores (torch.Tensor) : scores of shape (n_way * n_query, n_way)

        Notes:
            We initialize the fast parameters with the current parameters of the model, i.e., the so called
            slow parameters. We then used the fast parameters to compute the loss on the support set and
            compute the gradients of the loss with respect to the fast parameters. We then update the fast
            parameters with the gradients. We repeat this process for task_update_num iterations. Finally,
            we use the fast parameters to compute the loss on the query set.
        """

        # Get fast parameters
        fast_parameters = list(self.parameters())

        # Reset the fast parameters
        for weight in self.parameters():
            weight.fast = None

        # Reset the gradients
        self.zero_grad()

        # Try to adapt the model to the given support set
        for _ in range(self.task_update_num):
            # Parse the input data: backbone + (SOT) + flatten
            x_support, x_query, y_support = self.parse_feature(x)

            # Compute the predictions and loss for the support set
            scores = self.forward(x_support)
            set_loss = self.loss_fn(scores, y_support)

            # Build full graph support gradient of gradient
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)

            # Create/Update the fast parameters
            # (note the '-' is not merely minus value, but to create a new weight.fast)
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                # Create the fast parameter
                if weight.fast is None:
                    weight.fast = weight - self.inner_lr * grad[k]

                # Update the fast parameter
                else:
                    weight.fast = weight.fast - self.inner_lr * grad[k]

                # Add the fast parameter to the list
                fast_parameters.append(weight.fast)

        # Compute the scores for the query set
        scores = self.forward(x_query)

        return scores

    def set_forward_adaptation(
        self, x: Union[List[torch.Tensor], torch.Tensor], is_feature: bool = False
    ):
        raise ValueError(
            "MAML performs further adapation simply by increasing task_upate_num."
        )

    def set_forward_loss(
        self, x: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """Compute the loss for the current task.

        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): input (list of) tensor(s)

        Returns:
            torch.Tensor: loss tensor
        """

        # Get the query labels
        y_query = self.get_episode_labels(self.n_query, enable_grad=True)

        # Compute the scores
        scores = self.set_forward(x)

        # Compute the cross entropy loss
        loss = self.loss_fn(scores, y_query)

        return loss

    def train_loop(self, epoch, train_loader, optimizer):
        """
        Train the model for one epoch. The training is done in MAML fashion.
        This mean that we perform the "slow" parameter update after the given number of tasks.
        For each task we perform the "fast" parameter update. This means we adapt the model
        to the given support set and then compute the loss on the query set using the adapted
        "fast" parameters.

        Args:
            epoch (int) : current epoch
            train_loader (torch.utils.data.DataLoader) : train data loader
            optimizer (torch.optim.Optimizer) : optimizer
        """

        # Setup tracking variables
        avg_loss = 0
        task_count = 0
        loss_all = []

        # Iterate over the episodes / tasks and update
        # the model parameters in MAML fashion
        num_episodes = len(train_loader)
        pbar = self.get_progress_bar(enumerate(train_loader), total=num_episodes)
        pbar.set_description(
            f"Training: Epoch {epoch:03d} | Episodes 000/{num_episodes:03d} | 0.0000"
        )
        for i, (x, _) in enumerate(train_loader):
            # Reset the gradients
            optimizer.zero_grad()

            # Setup the number of query samples
            self.set_nquery(x)

            # Check if the number of classes is correct
            n_way = x.size(0)
            assert (
                self.n_way == n_way
            ), f"MAML do not support way change, n_way is {self.n_way} but x.size(0) is {x.size(0)}"

            # Compute the loss on the query set after adaptation on the support set
            loss = self.set_forward_loss(x)

            # Update the tracking variables
            avg_loss = avg_loss + loss.item()
            loss_all.append(loss)
            task_count += 1

            # Perform the MAML params update after n_task tasks
            if task_count == self.n_task:
                # Backpropagate the loss
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()
                optimizer.step()

                # Reset the tracking variables
                task_count = 0
                loss_all = []

            # Log the loss
            self.log_training_progress(pbar, epoch, i, num_episodes, avg_loss)

        epoch_loss = avg_loss / num_episodes

        return epoch_loss

    def test_loop(self, test_loader: torch.utils.data.DataLoader):
        """
        Test the model on the given data loader.

        Args:
            test_loader (torch.utils.data.DataLoader) : test data loader

        Returns:
            acc_mean (float) : mean accuracy
            acc_ci (float) : confidence interval of the accuracy
            acc_std (float) : std of the accuracy
        """

        # Get sample
        x, _ = next(iter(test_loader))

        # Check if the number of classes is correct
        n_way = x.size(0)
        assert (
            self.n_way == n_way
        ), f"MAML do not support way change, n_way is {self.n_way} but x.size(0) is {x.size(0)}"

        return super().test_loop(test_loader)
