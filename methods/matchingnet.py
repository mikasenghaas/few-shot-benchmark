# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate
from utils.data_utils import one_hot

from typing import Union, List


class MatchingNet(MetaTemplate):
    def __init__(
        self,
        backbone: torch.nn.Module,
        n_way: int,
        n_support: int,
        embed_support: bool = True,
        embed_query: bool = True,
        **kwargs
    ):
        """
        MatchingNet implementation. TODO: add more explanation

        Args:
            backbone (torch.nn.Module) : backbone network for the method
            n_way (int) : number of classes in a task
            n_support (int) : number of support samples per class
            embed_support (bool) : whether to embed the support set
            embed_query (bool) : whether to embed the query set
        """

        # Init the parent class
        super(MatchingNet, self).__init__(backbone, n_way, n_support, **kwargs)

        # Define the negative log likelihood loss function (since we are outputting log probabilities)
        self.loss_fn = nn.NLLLoss()

        # Define the Fully Contextual Embedder which is used to obtain
        # the weighted sum of the support set embeddings for each query embedding
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Define the encoders for support and query
        self.embed_support = (
            nn.LSTM(self.feat_dim, self.feat_dim, 1, bidirectional=True)
            if embed_support
            else None
        )
        self.embed_query = (
            FullyContextualEmbedding(self.feat_dim) if embed_query else None
        )

        # Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward_support_lstm(self, x: torch.Tensor) -> torch.Tensor:
        """Re-encode an input tensor of support samples using an LSTM.

        Args:
            x (torch.Tensor): input tensor of shape (n_way * n_support, feat_dim)

        Returns:
            torch.Tensor: re-encoded tensor of shape (n_way * n_support, feat_dim)
        """
        assert x.ndim == 2, "Input tensor must be 2D. Call `reshape2feature()` first."

        # Re-embed the support set
        out, _ = self.embed_support(x)
        x = x + out[:, : self.feat_dim] + out[:, self.feat_dim :]

        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)  # L2 norm
        x_normalised = x.div(x_norm + 0.00001)  # 0.00001 is to avoid division by zero

        return x, x_normalised

    def forward_query_lstm(
        self, x_query: torch.Tensor, x_support: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the query set using an LSTM based on the support set.

        Args:
            x (torch.Tensor): input tensor of shape (n_way * n_query, feat_dim)

        """
        x = self.embed_query(x_query, x_support)

        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalised = x.div(x_norm + 0.00001)  # shape = (n_way * n_query, feat_dim)

        return x, x_normalised

    def set_forward(
        self,
        x: Union[List[torch.Tensor], torch.Tensor],
        return_intermediates: bool = False,
    ):
        """
        Args:
            x (Union[List[torch.Tensor], torch.Tensor]) : input data of shape (batch_size, *)
            is_feature (bool) : if True, x is the feature vector of the input data

        Returns:
            logprobs (torch.Tensor) : log probabilities of shape (n_way * n_query, n_way)
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

        # Flatten the support and query sets to be of shape (n_way * n_support, feat_dim)
        x_support = x_support.contiguous().view(self.n_way * self.n_support, -1)
        x_query = x_query.contiguous().view(self.n_way * self.n_query, -1)

        # Encode the support and query sets using an LSTM
        if self.embed_support:
            _, x_support = self.forward_support_lstm(x_support)
        if self.embed_query:
            _, x_query = self.forward_query_lstm(x_query, x_support)
        if self.embed_support or self.embed_query:
            if return_intermediates:
                outputs["lstm"] = self.reshape2set(torch.cat([x_support, x_query]))

        # Get the log probabilities for each class
        cos_similarity = x_query @ x_support.T
        probs = self.softmax(self.relu(cos_similarity) * 100)

        # Get the labels of the support set
        y_s = self.get_episode_labels(self.n_support, enable_grad=False)
        Y_s = Variable(one_hot(y_s, self.n_way))
        logprobs = ((probs @ Y_s) + 1e-6).log()

        outputs["scores"] = logprobs

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

        # Compute the scores (logprobs)
        outputs = self.set_forward(x)
        scores = outputs["scores"]

        # Compute the negative log likelihood loss
        loss = self.loss_fn(scores, y_query)

        return loss

    def cuda(self):
        super(MatchingNet, self).to(self.device)
        self.FCE = self.FCE.to(self.device)
        return self


class FullyContextualEmbedding(nn.Module):
    def __init__(self, feat_dim: int):
        """
        Fully Contextual Embedding module using an LSTMCell for few-shot learning.
        The LSTMCell is used to iteratively update the query embedding with the weighted sum of the support set embeddings.

        Args:
            feat_dim (int): feature dimension
        """

        # Init the parent class
        super(FullyContextualEmbedding, self).__init__()

        # Define the parameters
        self.lstmcell = nn.LSTMCell(feat_dim * 2, feat_dim)
        self.softmax = nn.Softmax(dim=1)
        self.c_0 = Variable(torch.zeros(1, feat_dim))
        self.feat_dim = feat_dim

        # Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, queries: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """
        Compute the fully contextual embedding for the query set.

        Args:
            queries (torch.Tensor): query set of shape (n_way * n_query, feat_dim)
            support (torch.Tensor): encoded support set of shape (n_way * n_support, feat_dim)

        Returns:
            torch.Tensor: fully contextual embedding of shape (n_way * n_query, feat_dim)
        """

        # Create the initial cell state (zero matrix with shape (n_way * n_query, feat_dim))
        initial_state = self.c_0.expand_as(queries)  # c = self.c_0.expand_as(f)
        hidden_state = queries

        logits = (
            queries @ support.T
        )  # logit_a = h.mm(G_T) (cosine similarity between query and support), (n_way * n_query, n_way * n_support)
        scores = self.softmax(logits)  # a = self.softmax(logit_a)
        queries_reweighted = (
            scores @ support
        )  # r = a.mm(G) (weighted sum of the support set embeddings), (n_way * n_query, feat_dim)

        # Stack the original query embeddings with the reweighted ones
        x = torch.cat((queries, queries_reweighted), 1)

        for _ in range(
            support.size(0)
        ):  # re-embed n_way * n_support times (seems to be a heuristic), performance generally increases with more iterations
            hidden_state, initial_state = self.lstmcell(
                x, (hidden_state, initial_state)
            )
            hidden_state = hidden_state + queries

        return hidden_state

    def cuda(self):
        super(FullyContextualEmbedding, self).to(self.device)
        self.c_0 = self.c_0.to(self.device)
        return self
