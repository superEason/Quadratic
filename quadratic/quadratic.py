import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init


class Identity(nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Quadratic(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Quadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wr = Parameter(torch.Tensor(out_features, in_features))
        self.wg = Parameter(torch.Tensor(out_features, in_features))
        self.wb = Parameter(torch.Tensor(out_features, in_features))
        self.bias = bias
        if bias:
            self.br = Parameter(torch.Tensor(out_features))
            self.bg = Parameter(torch.Tensor(out_features))
            self.b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.wr, a=math.sqrt(5))
        init.kaiming_uniform_(self.wg, a=math.sqrt(5))
        init.kaiming_uniform_(self.wb, a=math.sqrt(5))
        if self.bias is not None:
            fan_in_r, _ = init._calculate_fan_in_and_fan_out(self.wr)
            fan_in_g, _ = init._calculate_fan_in_and_fan_out(self.wg)
            fan_in_b, _ = init._calculate_fan_in_and_fan_out(self.wb)
            bound_r = 1 / math.sqrt(fan_in_r)
            bound_g = 1 / math.sqrt(fan_in_g)
            bound_b = 1 / math.sqrt(fan_in_b)
            init.uniform_(self.br, -bound_r, bound_r)
            init.uniform_(self.bg, -bound_g, bound_g)
            init.uniform_(self.b, -bound_b, bound_b)

    def forward(self, input):
        y1 = F.linear(input, self.wr, self.br)
        y2 = F.linear(input, self.wg, self.bg)
        y3 = F.linear(input**2, self.wb, self.b)
        return y1*y2+y3

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


