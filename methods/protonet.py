# This code is modified from https://github.com/jakesnell/prototypical-networks

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate

from typing import List, Tuple, Union

class ProtoNet(MetaTemplate):
    def __init__(
            self, 
            backbone : torch.nn.Module, 
            n_way : int, 
            n_support : int, 
            similarity : str = "euclidean",
            **kwargs
        ):
        """Protonet Meta Learner - compute the prototypes based on the support set and then
        return the chosen similarity measure between the prototypes and the query set.

        Args:
            backbone (torch.nn.Module): backbone network
            n_way (int): number of classes for each task
            n_support (int): number of support samples for each class
            similarity (str, optional): similarity metric to use. Defaults to "euclidean". 
        """

        # Init parent class
        super(ProtoNet, self).__init__(backbone, n_way, n_support, **kwargs)

        # Define Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        self.similarity_type = similarity

        # Define device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_similarity(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """Compute the similarity between queries and prototypes.

        Args:
            x (torch.Tensor): query tensor of shape (n_way * n_query, feat_dim)
            y (torch.Tensor): prototype tensor of shape (n_way, feat_dim)
        
        Returns:
            torch.Tensor: similarity tensor of shape (n_way * n_query, n_way)
        """

        if self.similarity_type == "euclidean":
            n = x.size(0)
            m = y.size(0)
            d = x.size(1)
            assert d == y.size(1)

            x = x.unsqueeze(1).expand(n, m, d)
            y = y.unsqueeze(0).expand(n, m, d)

            return torch.pow(x - y, 2).sum(2)
        else:
            raise NotImplementedError("Similarity type not implemented")

    def set_forward(self, x : Union[torch.Tensor, List[torch.Tensor]], is_feature : bool = False) -> torch.Tensor:
        """
        [MetaTraining] Compute the prototypes based on the support set and then
        return the inverted distance between the prototypes and the query set.
        (higher distance means lower score/similarity)

        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): input tensor
            is_feature (bool, optional): whether the input is feature or not. Defaults to False.
            Determines whether the backbone is run on the input or not.

        Returns:
            torch.Tensor: output tensor of shape (n_way * n_query, n_way) 
        """

        # Get the support and query embeddings
        z_support, z_query = self.parse_feature(x, is_feature)

        # Make sure the tensors are contiguous in the memory
        z_support = z_support.contiguous()

        # Get the prototypes for each class by averaging the support embeddings
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)

        # Turn query into shape (n_way * n_query, feat_dim)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # Compute the euclidean distance between the query and the prototypes
        dists = self.get_similarity(z_query, z_proto)

        # Compute the scores by inverting the distances
        # (the higher the distance, the lower the score)
        scores = -dists

        return scores

    def set_forward_loss(self, x : Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
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

        # Compute the loss
        loss = self.loss_fn(scores, y_query)

        return loss
