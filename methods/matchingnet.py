# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate
from utils.data_utils import one_hot

from typing import Tuple, Union, List


class MatchingNet(MetaTemplate):
    def __init__(self, backbone: torch.nn.Module, n_way: int, n_support: int, **kwargs):
        """
        MatchingNet implementation. TODO: add more explanation

        Args:
            backbone (torch.nn.Module) : backbone network for the method
            n_way (int) : number of classes in a task
            n_support (int) : number of support samples per class
        """

        # Init the parent class
        super(MatchingNet, self).__init__(backbone, n_way, n_support, **kwargs)

        # Define the negative log likelihood loss function (since we are outputting log probabilities)
        self.loss_fn = nn.NLLLoss()

        # Define the Fully Contextual Embedder which is used to obtain
        # the weighted sum of the support set embeddings for each query embedding
        self.FCE = FullyContextualEmbedding(self.feat_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Define the Encode of the support set
        self.G_encoder = nn.LSTM(
            self.feat_dim, self.feat_dim, 1, batch_first=True, bidirectional=True
        )

        # Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode_training_set(self, S: torch.Tensor, G_encoder: torch.nn.Module = None):
        """
        Encode the support set using the G_encoder (Default is LSTM)

        Args:
            S (torch.Tensor) : support set of shape (n_way * n_support, feat_dim)
            G_encoder (torch.nn.Module) : G encoder (Default is LSTM)

        Returns:
            G (torch.Tensor) : encoded support set of shape (n_way * n_support, feat_dim)
            G_normalized (torch.Tensor) : normalized encoded support set  of shape (n_way * n_support, feat_dim)
        """

        # Define the G_encoder if not defined
        if G_encoder is None:
            G_encoder = self.G_encoder

        # Obtain the last layer of the G_encoder
        # (the squeeze is to add/remove the batch dimension)
        out_G = G_encoder(S.unsqueeze(0))[0]
        # out_G shape = (n_way * n_support, feat_dim * 2)
        # (*2 since LSTM is bidirectional - we concat the forward and backward outputs)
        out_G = out_G.squeeze(0)

        # Obtain the G: keep the original features and add the results of the LSTM
        # from the forward and backward directions (think of it as a sort of residual connection)
        G = S + out_G[:, : S.size(1)] + out_G[:, S.size(1) :]

        # Normalize the G
        G_norm = torch.norm(G, p=2, dim=1).unsqueeze(1).expand_as(G)  # L2 norm
        G_normalized = G.div(G_norm + 0.00001)  # 0.00001 is to avoid division by zero

        return G, G_normalized

    def get_logprobs(
        self,
        z_query: torch.Tensor,
        G: torch.Tensor,
        G_normalized: torch.Tensor,
        Y_S: torch.Tensor,
        FCE: torch.nn.Module = None,
    ):
        """
        Get the log probabilities of the query set

        Args:
            z_query (torch.Tensor) : query set of shape (n_way * n_query, feat_dim)
            G (torch.Tensor) : encoded support set of shape (n_way * n_support, feat_dim)
            G_normalized (torch.Tensor) : normalized encoded support set  of shape (n_way * n_support, feat_dim)
            Y_S (torch.Tensor) : one-hot encoding of the support set labels of shape (n_way * n_support, n_way)
            FCE (torch.nn.Module) : fully contextual embedder

        Returns:
            logprobs (torch.Tensor) : log probabilities of shape (n_way * n_query, n_way)
        """

        # Define the Fully Contextual Embedder if not defined
        if FCE is None:
            FCE = self.FCE

        # Obtain the Normalised Fully Contextual Embedding of the query set
        # Shape of F = (n_way * n_query, feat_dim) --> each query embedding is now a weighted sum of the support set embeddings
        F = FCE(z_query, G)
        F_norm = (
            torch.norm(F, p=2, dim=1).unsqueeze(1).expand_as(F)
        )  # L2 norm of each embedding
        F_normalized = F.div(F_norm + 0.00001)  # shape = (n_way * n_query, feat_dim)

        # Obtain the scores of each class
        # First apply the dot product between the normalized query and support set embeddings: shape = (n_way * n_query, n_way * n_support)
        # Then apply the ReLU activation function and multiply by 100 to strengthen the highest probability after softmax
        scores = self.relu(F_normalized.mm(G_normalized.transpose(0, 1))) * 100

        # For each query embedding, obtain the importantce of each support embedding (we softmax along the rows)
        softmax = self.softmax(scores)

        # However, we want to obtain the importance of each class, not each support embedding
        # So we multiply the softmax scores by the one-hot encoding of the support set labels
        # Therefore we do (n_way * n_query, n_way * n_support) x (n_way * n_support, n_way) = (n_way * n_query, n_way)
        # Thus for each query and class, using the Y_S one-hot encoding, we obtain the importance of each class by only taking
        # the probability of the support set embeddings of that class
        logprobs = (softmax.mm(Y_S) + 1e-6).log()

        return logprobs

    def set_forward(
        self, x: Union[List[torch.Tensor], torch.Tensor], is_feature: bool = False
    ):
        """
        Args:
            x (Union[List[torch.Tensor], torch.Tensor]) : input data of shape (batch_size, *)
            is_feature (bool) : if True, x is the feature vector of the input data

        Returns:
            logprobs (torch.Tensor) : log probabilities of shape (n_way * n_query, n_way)
        """

        # Split the input data into support and query sets, if is_feature False -> run backbone
        # Shape of z_support = (n_way, n_support, feat_dim), similar for z_query
        z_support, z_query = self.parse_feature(x, is_feature)

        # Flatten the support and query sets to be of shape (n_way * n_support, feat_dim)
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # Encode the support set
        G, G_normalized = self.encode_training_set(z_support)

        # Get the labels of the support set
        y_s = self.get_episode_labels(
            self.n_support, enable_grad=False
        )  # shape = (n_way * n_support,)

        # Get the one-hot encoding of the support set labels, make sure gradients is enabled
        Y_S = Variable(one_hot(y_s, self.n_way)).to(
            self.device
        )  # shape = (n_way * n_support, n_way)

        # Get the log probabilities for each class
        logprobs = self.get_logprobs(
            z_query, G, G_normalized, Y_S
        )  # shape = (n_way * n_query, n_way)

        return logprobs

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
        scores = self.set_forward(x)

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

    def forward(self, f: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        Compute the fully contextual embedding for the query set.

        Args:
            f (torch.Tensor): query set of shape (n_way * n_query, feat_dim)
            G (torch.Tensor): encoded support set of shape (n_way * n_support, feat_dim)

        Returns:
            torch.Tensor: fully contextual embedding of shape (n_way * n_query, feat_dim)
        """
        h = f
        c = self.c_0.expand_as(f)
        G_T = G.transpose(0, 1)
        K = G.size(0)  # Tuna to be comfirmed
        for k in range(K):
            logit_a = h.mm(G_T)
            a = self.softmax(logit_a)
            r = a.mm(G)
            x = torch.cat((f, r), 1)

            h, c = self.lstmcell(x, (h, c))
            h = h + f

        return h

    def cuda(self):
        super(FullyContextualEmbedding, self).to(self.device)
        self.c_0 = self.c_0.to(self.device)
        return self
