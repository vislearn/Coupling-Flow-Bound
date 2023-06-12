from typing import Union

import torch
from FrEIA.modules import InvertibleModule
from torch import nn


class AxisFixedLinearTransform(InvertibleModule):
    def __init__(self, dims_in, dims_c=None, M: torch.Tensor = None,
                 b: Union[None, torch.Tensor] = None):
        '''Additional args in docstring of base class FrEIA.modules.InvertibleModule.

        Args:
          M: Square, invertible matrix, with which each input is multiplied. Shape ``(d, d)``.
          b: Optional vector which is added element-wise. Shape ``(d,)``.
        '''
        super().__init__(dims_in, dims_c)

        if M is None:
            raise ValueError(
                "Need to specify the M argument, the matrix to be multiplied.")

        self.M = nn.Parameter(M.t(), requires_grad=False)
        self.M_inv = nn.Parameter(M.t().inverse(), requires_grad=False)

        if b is None:
            self.b = 0.
        else:
            self.b = nn.Parameter(b.unsqueeze(0), requires_grad=False)

        self.logDetM = nn.Parameter(torch.slogdet(M)[1], requires_grad=False)

    def _mm(self, x):
        return torch.einsum("ij,bi...->bj...", self.M, x).contiguous()

    def forward(self, x, rev=False, jac=True):
        j = self.logDetM
        if not rev:
            out = self._mm(x[0]) + self.b
            return (out,), j
        else:
            out = self._mm(x[0] - self.b)
            return (out,), -j

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        return input_dims
