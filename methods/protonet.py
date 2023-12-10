# This code is modified from https://github.com/jakesnell/prototypical-networks

import torch
import torch.nn as nn

from methods.meta_template import MetaTemplate

from typing import List, Union


class ProtoNet(MetaTemplate):
    def __init__(
        self,
        backbone: torch.nn.Module,
        n_way: int,
        n_support: int,
        similarity: str = "euclidean",
        **kwargs,
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

        self.encoder = nn.LSTM(self.feat_dim, self.feat_dim, 1, bidirectional=True)

        # Define device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

    def forward_lstm(self, x: torch.Tensor) -> torch.Tensor:
        """Re-encode an input tensor using an LSTM. Is used to re-encode the support set,
        query set or both.

        Args:
            x (torch.Tensor): input tensor of shape (n_way * n_query, feat_dim)

        Returns:
            torch.Tensor: re-encoded tensor of shape (n_way * n_query, feat_dim)
        """
        assert x.ndim == 2, "Input tensor must be 2D. Call `reshape2feature()` first."

        # Re-embed the support set
        out, _ = self.encoder(x)
        x = x + out[:, : self.feat_dim] + out[:, self.feat_dim :]

        return x

    def set_forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        [MetaTraining] Compute the prototypes based on the support set and then
        return the inverted distance between the prototypes and the query set.
        (higher distance means lower score/similarity)

        Args:
            x (Union[torch.Tensor, List[torch.Tensor]]): input tensor
            return_intermediates (bool, optional): whether to save the intermediate tensors during the forward pass. Defaults to False.

        Returns:
            output: dict containing the scores and optionally the intermediate tensors
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

        # Run LSTM embedder
        x_support = self.forward_lstm(x_support)
        if return_intermediates:
            outputs["lstm"] = self.reshape2set(torch.cat([x_support, x_query]))

        # Get the prototypes for each class by averaging the support embeddings
        proto = x_support.view(self.n_way, self.n_support, -1).mean(1)

        # Turn query into shape (n_way * n_query, feat_dim)
        x_query = x_query.contiguous().view(self.n_way * self.n_query, -1)

        # Compute the euclidean distance between the query and the prototypes
        dists = self.get_similarity(x_query, proto)

        # Compute the scores by inverting the distances
        scores = -dists
        outputs["scores"] = scores

        return outputs

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
        outputs = self.set_forward(x)
        scores = outputs["scores"]

        # Compute the loss
        loss = self.loss_fn(scores, y_query)

        return loss
