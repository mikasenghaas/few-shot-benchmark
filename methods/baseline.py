import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import wandb
from abc import ABC
from IPython.display import clear_output

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
        backbone : nn.Module,
        n_way : int,
        n_support : int,
        n_classes : int = 1,
        loss : str = "softmax",
        type : str = "classification",
        log_wandb : bool = True,
        print_freq : int = 10,
    ):

        # Initialize the the MetaTemplate parent class
        change_way = True
        super(Baseline, self).__init__(backbone, n_way, n_support, change_way, log_wandb, print_freq, type)

        # Define the feature extractor
        self.feature = backbone

        # Define the type of the task (classification or regression)
        self.type = type

        # Define the number of classes (for classification) which defines the output dimension
        # of the classifier
        self.n_classes = n_classes

        # Define the classifier
        if loss == "softmax":
            self.classifier = nn.Linear(self.feature.final_feat_dim, n_classes)
            self.classifier.bias.data.fill_(0)
        elif loss == "dist":  # Baseline ++
            self.classifier = distLinear(self.feature.final_feat_dim, n_classes)
        self.loss_type = loss  # 'softmax' #'dist'

        # Define the loss function based on the task
        if self.type == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.type == "regression":
            self.loss_fn = nn.MSELoss()

        # Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        [Pretraining] Forward the data through the backbone.

        Args:
            x (torch.Tensor / [torch.Tensor]): the input data

        Returns:
            out (torch.Tensor): the output of the backbone
        """

        # TODO: why list?
        if isinstance(x, list):
            x = [Variable(obj.to(self.device)) for obj in x]
        
        # Turn the input data into a Variable so we can compute the gradient
        else:
            x = Variable(x.to(self.device))

        # Extract the features
        out = self.feature.forward(x)

        # Using chosen classifer, map the embeddings vector for each to the number of classes
        if self.classifier != None:
            scores = self.classifier.forward(out)
        else:
            raise ValueError("Classifier not defined, Regression not supported.")

        return scores

    def set_forward_loss(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        [Pretraining] Forward the data through the model and compute the loss.

        Args:
            x (torch.Tensor / [torch.Tensor]): the input data
            y (torch.Tensor): the ground truth
        
        Returns:
            loss (torch.Tensor): the loss
        """

        # Compute the prediction
        scores = self.forward(x)

        # Get the labels and cast it to correct type
        if self.type == "classification":
            y = y.long().to(self.device)
        else:
            y = y.to(self.device)

        # Compute the loss and return
        return self.loss_fn(scores, y)

    def train_loop(self, epoch : int, train_loader : torch.utils.data.DataLoader, optimizer : torch.optim.Optimizer) -> float:
        """
        [Pretraining] Train the model for one epoch and log the loss.

        Args:
            epoch (int): the current epoch
            train_loader (DataLoader): the training data loader
            optimizer (Optimizer): the optimizer
        
        Returns:
            avg_loss (float): the average loss over the epoch
        """

        # Initialize tracking variables
        avg_loss = 0

        # Train loop
        for i, (x, y) in enumerate(train_loader):
            # Forward the data through the model and compute the loss
            optimizer.zero_grad()
            loss = self.set_forward_loss(x, y)

            # Backpropagate the loss
            loss.backward()
            optimizer.step()

            # Log the loss
            avg_loss = avg_loss + loss.item()

            # Print the loss
            if i % self.print_freq == 0:
                clear_output(wait=True)
                current_loss = avg_loss / float(i + 1)
                print(
                    "ℹ️ Epoch {:d} | Batch {:d}/{:d} | Loss {:f}".format(
                        epoch, i, len(train_loader), current_loss
                    )
                )
                if self.log_wandb: wandb.log({"loss/train": current_loss})

        return avg_loss / len(train_loader)

    def set_forward(self, x : torch.Tensor, y : torch.Tensor = None) -> torch.Tensor:
        """
        [Finetuning] Fine-tune the model for the given episode's data
        and then evaluate the model on the query set.
        
        Args:
            x (torch.Tensor / [torch.Tensor]): the input data
            y (torch.Tensor): the ground truth 

        Returns:
            scores (torch.Tensor): the predictions on the query set
        """

        # Run backbone, split data into support and query sets
        z_support, z_query = self.parse_feature(x, is_feature=False)

        # Freeze the backbone
        z_support = (
            z_support.contiguous()
            .view(self.n_way * self.n_support, -1)
            .detach()
            .to(self.device)
        )
        z_query = (
            z_query.contiguous()
            .view(self.n_way * self.n_query, -1)
            .detach()
            .to(self.device)
        )

        # Classification
        if y is None:
            y_support = self.get_support_labels()
        
        # Regression
        else:  
            raise ValueError("Finetuning baseline on regression not supported.")
        
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
            linear_clf, set_optimizer, (z_support, y_support), z_query, loss_function, batch_size=4
        )

        return scores