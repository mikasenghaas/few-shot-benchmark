import numpy as np
import torch
import torch.nn as nn
import wandb
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
    """
    def __init__(
        self,
        backbone,
        n_way,
        n_support,
        n_classes=1,
        loss="softmax",
        type="classification",
    ):

        # Initialize the the MetaTemplate parent class
        super(Baseline, self).__init__(backbone, n_way, n_support, change_way=True)

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

    def forward(self, x):
        """
        Forward the data through the backbone.

        Args:
            x (torch.Tensor / [torch.Tensor]): the input data
        """

        # TODO: why list?
        if isinstance(x, list):
            x = [Variable(obj.to(self.device)) for obj in x]
        
        # Turn the input data into a Variable so we can compute the gradient
        else:
            x = Variable(x.to(self.device))

        # Extract the features
        out = self.feature.forward(x)

        # Using Linear layer, map the features to the output dimension
        if self.classifier != None:
            scores = self.classifier.forward(out)
        else:
            raise ValueError("Classifier not defined, Regression not supported.")

        return scores

    def set_forward_loss(self, x, y):
        """
        Forward the data through the model and compute the loss.

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

    def train_loop(self, epoch, train_loader, optimizer):
        """
        Train the model for one epoch and log the loss.

        Args:
            epoch (int): the current epoch
            train_loader (DataLoader): the training data loader
            optimizer (Optimizer): the optimizer
        
        Notes:
            We override the train_loop method from the parent class because
            TODO: why?
        """

        # Define the print frequency and initialize the average loss
        print_freq = 10
        avg_loss = 0

        # Train loop
        print("here")
        for i, (x, y) in enumerate(train_loader):
            print("got the data")
            # Forward the data through the model and compute the loss
            optimizer.zero_grad()
            loss = self.set_forward_loss(x, y)

            # Backpropagate the loss
            loss.backward()
            optimizer.step()

            # Log the loss
            avg_loss = avg_loss + loss.item()

            # Print the loss
            if i % print_freq == 0:
                print(
                    "Epoch {:d} | Batch {:d}/{:d} | Loss {:f}".format(
                        epoch, i, len(train_loader), avg_loss / float(i + 1)
                    )
                )
                wandb.log({"loss/train": avg_loss / float(i + 1)})

    def test_loop(self, test_loader, return_std=None):
        """
        Test the model and log the accuracy.

        Args:
            test_loader (DataLoader): the test data loader
            return_std (bool): whether to return the standard deviation
        
        Returns:
            acc_mean (float): the mean accuracy
            acc_std (float): the standard deviation of the accuracy
        
        Notes:
            We override the test_loop method from the parent class because
            TODO: why?
        """

        # Collect the accuracy for each class, TODO: is this true, each class?
        acc_all = []
        iter_num = len(test_loader)
        for x, y in test_loader:

            # TODO: not sure why we have to do this since we do not use it
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
            else:
                self.n_query = x.size(1) - self.n_support

            # Compute the accuracy
            if self.type == "classification":
                correct_this, count_this = self.correct(x)
                acc_all.append(correct_this / count_this * 100)
            # Compute the pearson correlation
            else:
                acc_all.append(self.correlation(x, y))

        # Compute the mean and standard deviation of the accuracy
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

        # Print the results
        if self.type == "classification":
            print(
                "%d Accuracy = %4.2f%% +- %4.2f%%"
                % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
            )
        else:
            # print correlation
            print(
                "%d Correlation = %4.2f +- %4.2f"
                % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
            )

        # Return the mean and (standard deviation) of the accuracy
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def set_forward(self, x, y=None):
        """
        Full training loop.

        TODO: provide more in depth explanation.
        TODO: why are we not using train loop function here?
        
        Args:
            x (torch.Tensor / [torch.Tensor]): the input data
            y (torch.Tensor): the ground truth 

        Returns:
            scores (torch.Tensor): the scores 
        """

        # Run backbone on the input data and then split 
        # the output into support and query sets
        # Shape of z_support: [n_way * n_support, feat_dim]
        # Shape of z_query: [n_way * n_query, feat_dim]
        z_support, z_query = self.parse_feature(x, is_feature=False)

        # Detach ensures we don't change the weights in main training process
        # TODO: why do need the weights to be part of the main training process?
        # (I guess because we only want to train the classifier?)
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
            # Example:
            # np.repeat([0, 1, 2], 2) --> [0, 0, 1, 1, 2, 2]
            y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
            y_support = Variable(y_support.to(self.device))
        
        # Regression
        else:  
            y_support = y[:, : self.n_support]
            y_support = (
                y_support.contiguous()
                .view(self.n_way * self.n_support, -1)
                .to(self.device)
            )
        
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

        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()

                # Get random selection for the current batch / episode
                selected_id = torch.from_numpy(
                    rand_id[i : min(i + batch_size, support_size)]
                ).to(self.device)

                # Get the support set for the current batch / episode
                z_batch = z_support[selected_id]

                # Get the labels for the support set for the current batch / episode
                y_batch = y_support[selected_id]

                # Compute the scores and loss
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)

                # Backpropagate the loss
                loss.backward()
                set_optimizer.step()

        # Compute the final scores for the query set 
        scores = linear_clf(z_query)

        return scores
