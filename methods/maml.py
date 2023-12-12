# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml
from typing import List, Union
import torch
import torch.nn as nn

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

    def forward(self, x, return_intermediates: bool = False):
        """
        Forwards an input of support and query samples through the model.
        Runs the backbone, SOT (if specified) and the classifier on the input.

        Args:
            x (torch.Tensor) : input tensor of shape (batch_size, *)
            return_intermediates (bool) : whether to return the intermediate features

        Returns:
            outputs (dict) : dictionary containing the outputs of the model
        """
        # Initialise outputs dict
        outputs = {}

        x = self.reshape2feature(x)
        if return_intermediates:
            outputs["input"] = self.reshape2set(x)

        # Run backbone
        x = self.forward_backbone(x)
        if return_intermediates:
            outputs["backbone"] = self.reshape2set(x)

        # Run SOT if specified
        if self.sot:
            x = self.forward_sot(x)
            if return_intermediates:
                outputs["sot"] = self.reshape2set(x)

        # Run classification head
        scores = self.classifier.forward(x)

        # Split and reshape the scores
        scores_support, scores_query = self.parse_feature(self.reshape2set(scores))
        scores_support = scores_support.contiguous().view(
            self.n_way * self.n_support, -1
        )
        scores_query = scores_query.contiguous().view(self.n_way * self.n_query, -1)

        outputs["scores_support"] = scores_support
        outputs["scores"] = scores_query

        return outputs

    def set_forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Run backbone and classifier on input data and perform adaptation.

        Args
            x (Union[List[torch.Tensor], torch.Tensor]) : input data of shape (batch_size, *)
            return_intermediates (bool): whether to return the intermediate features

        Returns:
            outputs (dict) : dictionary containing the outputs of the model

        Notes:
            We initialize the fast parameters with the current parameters of the model, i.e., the so called
            slow parameters. We then used the fast parameters to compute the loss on the support set and
            compute the gradients of the loss with respect to the fast parameters. We then update the fast
            parameters with the gradients. We repeat this process for task_update_num iterations. Finally,
            we use the fast parameters to compute the loss on the query set.
        """

        # Dynamically set the number of support samples
        self.set_nquery(x)

        # Get fast parameters
        fast_parameters = list(self.parameters())

        # Reset the fast parameters
        for weight in self.parameters():
            weight.fast = None

        # Reset the gradients
        self.zero_grad()

        # Get the labels of the support set
        y_support = self.get_episode_labels(self.n_support, enable_grad=True)

        # Try to adapt the model to the given support set
        for _ in range(self.task_update_num):
            # Forward the the episode through the model (extract support scores)
            outputs = self.forward(x)
            scores = outputs["scores_support"]

            # Compute the predictions and loss for the support set
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
        outputs = self.forward(x, return_intermediates=return_intermediates)

        return outputs

    def set_forward_adaptation(
        self, x: Union[List[torch.Tensor], torch.Tensor], is_feature: bool = False
    ):
        raise ValueError(
            "MAML performs further adapation simply by increasing task_upate_num."
        )

    def set_forward_loss(
        self, x: Union[torch.Tensor, List[torch.Tensor]], y
    ) -> torch.Tensor:
        """Compute the loss for the current task.

        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): input (list of) tensor(s)

        Returns:
            torch.Tensor: loss tensor
        """

        # Compute the scores
        outputs = self.set_forward(x)
        scores = outputs["scores"]

        # Get the query labels
        # y_query = self.get_episode_labels(self.n_query, enable_grad=True)
        y_query = y[:, self.n_support :].reshape(self.n_way * self.n_query).long()

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
        loss = 0
        task_count = 0
        loss_all = []

        # Iterate over the episodes / tasks and update
        # the model parameters in MAML fashion
        num_episodes = len(train_loader)
        pbar = self.get_progress_bar(enumerate(train_loader), total=num_episodes)
        pbar.set_description(
            f"Training: Epoch {epoch:03d} | Episodes 000/{num_episodes:03d} | 0.0000"
        )
        for i, (x, y) in pbar:
            # Reset the gradients
            optimizer.zero_grad()

            self.set_nquery(x)

            # Reshuffle
            x, y = self.shuffle_queries(x, y)

            # Integer encode
            mapper = {c.item(): i for i, c in enumerate(y[:, 0])}
            y = y.apply_(lambda x: mapper[x])

            # Check if the number of classes is correct
            n_way = x.size(0)
            assert (
                self.n_way == n_way
            ), f"MAML do not support way change, n_way is {self.n_way} but x.size(0) is {x.size(0)}"

            # Compute the loss on the query set after adaptation on the support set
            episode_loss = self.set_forward_loss(x, y)

            # Update the tracking variables
            loss += episode_loss.item()
            loss_all.append(episode_loss)
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
            self.log_training_progress(pbar, epoch, i, num_episodes, loss)

        epoch_loss = loss / num_episodes

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
