# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    softmax,
    add_remaining_self_loops,
)
from torch_geometric.nn.conv import MessagePassing


# Basic ResNet model


# %%
def init_layer(L):
    """
    Initialize a Conv1d, Conv2d or BatchNorm2d layer.

    :param L: Layer to be initialized. `L.kernel_size` is expected to be at least two-dimensional.

    Based on the layer type, the initialization is done as follows:

    - **Conv1d/Conv2d**: Initialize with Kaiming He normal initialization
    - **BatchNorm2d**: Initialize with ones for weights and zeros for biases.
    """
    if isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)
        return

    assert len(L.kernel_size) >= 2, "Expected kernel size to be at least 2D"

    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
        return

    if isinstance(L, nn.Conv1d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
        return


class distLinear(nn.Module):
    """
    This class implements a specialized linear layer where both inputs and weights are normalized. The normalization ensures that the dot product between them computes a **cosine similarity**, scaled by `self.scale_factor`. Cosine similarity for two vectors :math:`A` and :math:`B` is defined as:

    :math:`similarity = A \\cdot B / (||A|| * ||B||)`

    where :math:`||A||` is the L2 norm of :math:`A`.
    """

    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True  # See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(
                self.L, "weight", dim=0
            )  # split the weight update component to direction and norm

        if outdim <= 200:
            self.scale_factor = 2
            # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            self.scale_factor = 10
            # in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(
            x_norm + 0.00001
        )  # divides by norm of row vectors of x, handles 0 norm cases
        if not self.class_wise_learnable_norm:
            L_norm = (
                torch.norm(
                    self.L.weight.data, p=2, dim=1
                )  # compute the norm of each row vector of L.weight.data
                .unsqueeze(1)
                .expand_as(self.L.weight.data)
            )
            self.L.weight.data = self.L.weight.data.div(
                L_norm + 0.00001
            )  # normalize the row vectors of L.weight.data same as with x_normalized
        cos_dist = self.L(
            x_normalized
        )  # matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor * (cos_dist)

        return scores


class Flatten(nn.Module):
    """
    This class flattens the input tensor to 2D, where the batch dimension is preserved while the other dimensions are flattened.
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Linear_fw(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        """
        This class implements a linear layer where the weights can be replaced by a fast weight during forward pass.
        This is used in inner loop of MAML to forward input with fast weight (temporary parameters which will be used to update the original parameters).

        Args:
            in_features: Number of input features
            out_features: Number of output features

        """
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        """
        Uses fast weights if they are defined. Otherwise, uses the normal nn.Linear layer.
        """
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(
                x, self.weight.fast, self.bias.fast
            )  # weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)  # use the normal linear layer
        return out


class Conv1d_fw(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = True,
    ):
        """
        This class implements a 1D convolutional layer where the weights can be replaced by a fast weight during forward pass.
        This is used in inner loop of MAML to forward input with fast weight (temporary parameters which will be used to update the original parameters).
        """
        super(Conv1d_fw, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        """
        Uses fast weights if they are defined. Otherwise, uses the normal nn.Conv1d layer.
        """
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv1d(
                    x, self.weight.fast, None, stride=self.stride, padding=self.padding
                )
            else:
                out = super(Conv1d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv1d(
                    x,
                    self.weight.fast,
                    self.bias.fast,
                    stride=self.stride,
                    padding=self.padding,
                )
            else:
                out = super(Conv1d_fw, self).forward(x)

        return out


class Conv2d_fw(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = True,
    ):
        """
        This class implements a 2D convolutional layer where the weights can be replaced by a fast weight during forward pass.
        This is used in inner loop of MAML to forward input with fast weight (temporary parameters which will be used to update the original parameters).
        """
        super(Conv2d_fw, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """
        Uses fast weights if they are defined. Otherwise, uses the normal nn.Conv2d layer.
        """
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(
                    x, self.weight.fast, None, stride=self.stride, padding=self.padding
                )
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(
                    x,
                    self.weight.fast,
                    self.bias.fast,
                    stride=self.stride,
                    padding=self.padding,
                )
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features: int):
        """
        This class implements a batch normalization layer where the weights can be replaced by a fast weight during forward pass.
        This is used in inner loop of MAML to forward input with fast weight (temporary parameters which will be used to update the original parameters).
        """
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        """
        Uses fast weights if they are defined. Otherwise, uses the normal nn.BatchNorm2d layer.
        """
        running_mean = torch.zeros(x.data.size()[1]).to(self.device)
        running_var = torch.ones(x.data.size()[1]).to(self.device)
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight.fast,
                self.bias.fast,
                training=True,
                momentum=1,
            )
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                training=True,
                momentum=1,
            )
        return out


class BatchNorm1d_fw(nn.BatchNorm1d):
    def __init__(self, num_features: int):
        """
        This class implements a batch normalization layer where the weights can be replaced by a fast weight during forward pass.
        This is used in inner loop of MAML to forward input with fast weight (temporary parameters which will be used to update the original parameters).
        """
        super(BatchNorm1d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """
        Uses fast weights if they are defined. Otherwise, uses the normal nn.BatchNorm1d layer.
        """
        running_mean = torch.zeros(x.data.size()[1]).to(self.device)
        running_var = torch.ones(x.data.size()[1]).to(self.device)
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight.fast,
                self.bias.fast,
                training=True,
                momentum=1,
            )
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                training=True,
                momentum=1,
            )
        return out


class LayerNorm_fw(nn.LayerNorm):  # layer norm MAML attempt
    def __init__(self, num_features, elementwise_affine=True):
        super(LayerNorm_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(num_features), requires_grad=True)
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters to their initialization values.
        """
        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.layer_norm(
                x, self.normalized_shape, self.weight.fast, self.bias.fast
            )
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.layer_norm(x, self.normalized_shape, self.weight, self.bias)
        return out


# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, convdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = BatchNorm2d_fw(outdim)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim: int, outdim: int, half_res: bool):
        """
        Simple Block used in ResNet. Uses two 3x3 convolutions and a shortcut connection to implement identity mapping.


        Args:
            indim: Number of input channels
            outdim: Number of output channels
            half_res: If True, then the shortcut connection downsamples the input by a factor of 2. Otherwise, the shortcut connection preserves the input resolution.
        """
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(
                indim,
                outdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1,
                bias=False,
            )
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(
                indim,
                outdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1,
                bias=False,
            )
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(
                    indim, outdim, 1, 2 if half_res else 1, bias=False
                )
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(
                    indim, outdim, 1, 2 if half_res else 1, bias=False
                )
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = "1x1"
        else:
            self.shortcut_type = "identity"

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        """
        Standard implementation of ResNet block. Idea behind ResNet is to use identity shortcut connections to avoid vanishing gradients.
        """
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = (
            x if self.shortcut_type == "identity" else self.BNshortcut(self.shortcut(x))
        )
        out = out + short_out
        out = self.relu2(out)
        return out


# Bottleneck block
class BottleneckBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim / 4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1, bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(
                bottleneckdim,
                bottleneckdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1,
            )
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1, bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(
                bottleneckdim,
                bottleneckdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1,
            )
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [
            self.C1,
            self.BN1,
            self.C2,
            self.BN2,
            self.C3,
            self.BN3,
        ]
        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(
                    indim, outdim, 1, stride=2 if half_res else 1, bias=False
                )
            else:
                self.shortcut = nn.Conv2d(
                    indim, outdim, 1, stride=2 if half_res else 1, bias=False
                )

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = "1x1"
        else:
            self.shortcut_type = "identity"

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        short_out = x if self.shortcut_type == "identity" else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


def full_block(in_features: int, out_features: int, dropout: float):
    """
    A fully connected block used in FCNet.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        dropout: Dropout probability

    Returns:
        A Sequential module consisting of a linear layer, batch normalization, ReLU activation and dropout in that order
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(p=dropout),
    )


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def full_block_fw(in_features, out_features, dropout):
    """
    A fully connected block used in FCNet with fast weights.
    Used in inner loop of MAML to forward input with fast weight (temporary parameters which will be used to update the original parameters).

    Args:
        in_features: Number of input features
        out_features: Number of output features
        dropout: Dropout probability

    Returns:
        A Sequential module consisting of a linear layer (with fast weights), batch normalization (with fast weights), ReLU activation and dropout in that order
    """
    return nn.Sequential(
        Linear_fw(in_features, out_features),  # use weight fast weight in forward pass
        BatchNorm1d_fw(out_features),  # use weight fast weight in forward pass
        nn.ReLU(),
        nn.Dropout(p=dropout),
    )
