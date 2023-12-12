from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from backbones.blocks import distLinear
from methods.meta_template import MetaTemplate


class Baseline(MetaTemplate):
    """
    Baseline model for few-shot classification

    Args:
        backbone (nn.Module): the feature extractor
        n_way (int): number of classes in a classification task
        n_support (int): number of support examples per class in the support set
        n_classes (int): number of classes in the training set
        loss (str): the loss function to be used (softmax or dist)
        type (str): the type of the task (classification or regression)
        log_wandb (bool): whether to log the results to wandb
        print_freq (int): how often (in terms of # of batches) to print the results
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_way: int,
        n_support: int,
        n_classes: int = 1,
        loss: str = "softmax",
        type: str = "classification",
        **kwargs,
    ):
        # Initialize the the MetaTemplate parent class
        change_way = True
        super(Baseline, self).__init__(
            backbone, n_way, n_support, change_way, type=type, **kwargs
        )

        # Define the feature extractor
        self.feature = backbone

        # Define the type of the task (classification or regression)
        self.type = type

        # Define the number of classes (for classification) which defines the output dimension
        # of the classifier
        self.n_classes = n_classes

        # Define the classifier used only the training phase!
        if loss == "softmax":
            self.classifier = nn.Linear(self.feat_dim, n_classes)
            self.classifier.bias.data.fill_(0)
        elif loss == "dist":  # Baseline ++
            self.classifier = distLinear(self.feat_dim, n_classes)
        self.loss_type = loss  # 'softmax' #'dist'

        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        [Pretraining] Forward the data through the backbone.

        Args:
            x (torch.Tensor / [torch.Tensor]): the input data
            return_intermediates (bool): whether to return the intermediate features

        Returns:
            out (dict): the output of the model
        """

        # Initialise outputs dict
        outputs = {}

        # Extract the features
        x = self.forward_backbone(x)
        if return_intermediates:
            outputs["backbone"] = x

        if self.sot:
            x = self.sot(x)
            if return_intermediates:
                outputs["sot"] = x

        scores = self.classifier.forward(x)
        outputs["scores"] = scores

        return outputs

    def set_forward_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        [Pretraining] Forward the data through the model and compute the loss.

        Args:
            x (torch.Tensor / [torch.Tensor]): the input data
            y (torch.Tensor): the ground truth

        Returns:
            loss (torch.Tensor): the loss
        """

        # Compute the prediction
        outputs = self.forward(x)
        scores = outputs["scores"]

        # Get the labels and cast it to correct type
        y = y.long().to(self.device)

        # Compute the loss and return
        return self.loss_fn(scores, y)

    def train_loop(
        self,
        epoch: int,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        [Pretraining] Train the model for one epoch and log the loss.

        Args:
            epoch (int): the current epoch
            train_loader (DataLoader): the training data loader
            optimizer (Optimizer): the optimizer

        Returns:
            avg_loss (float): the average loss over the epoch
        """

        # Train loop
        num_batches = len(train_loader)
        pbar = self.get_progress_bar(enumerate(train_loader), total=num_batches)
        pbar.set_description(
            f"Epoch {epoch:03d} | Batch 000/{num_batches:03d} | 0.0000"
        )
        loss = 0.0
        for i, (x, y) in pbar:
            # Forward the data through the model and compute the loss
            optimizer.zero_grad()
            batch_loss = self.set_forward_loss(x, y)

            # Backpropagate the loss
            batch_loss.backward()
            optimizer.step()

            # Log the loss
            loss += batch_loss.item()

            # Print the loss
            self.log_training_progress(
                pbar, epoch, i, num_batches, loss, few_shot=False
            )

        # Compute the average loss
        epoch_loss = loss / num_batches

        return epoch_loss

    def set_forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        [Finetuning] Fine-tune the model for the given episode's data
        and then evaluate the model on the query set.

        Args:
            x (torch.Tensor / [torch.Tensor]): the input data
            return_intermediates (bool): whether to return the intermediate features

        Returns:
            scores (torch.Tensor): the predictions on the query set
        """

        # Set the number of query samples dynamically
        self.set_nway(x)
        self.set_nquery(x)

        # Initialise outputs dict
        outputs = {}

        # Reshape input to 2d tensor of shape (n_way * (n_support + n_query), feat_dim)
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

        # Split support and query
        x_support, x_query = self.parse_feature(self.reshape2set(x))

        # Freeze the backbone
        x_support = (
            x_support.contiguous()
            .view(self.n_way * self.n_support, -1)
            .detach()
            .to(self.device)
        )
        x_query = (
            x_query.contiguous()
            .view(self.n_way * self.n_query, -1)
            .detach()
            .to(self.device)
        )

        # Get episode labels
        y_support = self.get_episode_labels(self.n_support, enable_grad=True)

        # Define classifier
        if self.loss_type == "softmax":
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == "dist":
            linear_clf = distLinear(self.feat_dim, self.n_way)
        else:
            raise ValueError("Loss type not supported")

        linear_clf = linear_clf.to(self.device)

        # Define optimizer
        set_optimizer = torch.optim.SGD(
            linear_clf.parameters(),
            lr=0.01,
            momentum=0.9,
            dampening=0.9,
            weight_decay=0.001,
        )

        loss_function = self.loss_fn.to(self.device)

        # Finetune the classifier
        scores = self.adapt(
            linear_clf,
            set_optimizer,
            (x_support, y_support),
            x_query,
            loss_function,
            batch_size=4,
        )

        outputs["scores"] = scores

        return outputs

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
