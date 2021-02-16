"""NICE With Virtual Bottle Neck Compression model
"""

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform, SigmoidTransform, AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np
"""Additive coupling layer.
"""
class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()
        # configure coupling function m
        assert hidden >= 0
        self.m = nn.Sequential(
            nn.Linear(in_out_dim, mid_dim),
            nn.LeakyReLU(0.1),
            *sum(([nn.Linear(mid_dim, mid_dim), nn.LeakyReLU(0.1)] for _ in range(hidden)), []),
            nn.Linear(mid_dim, in_out_dim)
        )
        # configure mask
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert mask_config in {0, 1}
        self.mask = torch.zeros(in_out_dim, device=device) if not mask_config else torch.ones(in_out_dim, device=device)
        self.mask[::2] = 1 - mask_config


    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        if not reverse:
            x1, x2 = self.mask * x, (1. - self.mask) * x
            y1, y2 = x1, x2 + (self.m(x1) * (1. - self.mask))
            return y1 + y2, log_det_J

        y1, y2 = self.mask * x, (1. - self.mask) * x
        x1, x2 = y1, y2 - (self.m(y1) * (1. - self.mask))
        return x1 + x2, log_det_J

class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        # configure coupling function m
        assert hidden >= 0
        self.s = nn.Sequential(
            nn.Linear(in_out_dim, mid_dim),
            nn.LeakyReLU(0.05),
            *sum(([nn.Linear(mid_dim, mid_dim), nn.LeakyReLU(0.05), nn.BatchNorm1d(mid_dim)] for _ in range(hidden)), []),
            nn.Linear(mid_dim, in_out_dim),
            nn.BatchNorm1d(in_out_dim),
            nn.Tanh(), # to avoid from exponent explosion see
        )
        self.t = nn.Sequential(
            nn.Linear(in_out_dim, mid_dim),
            nn.LeakyReLU(0.05),
            *sum(([nn.Linear(mid_dim, mid_dim), nn.LeakyReLU(0.05), nn.BatchNorm1d(mid_dim)] for _ in range(hidden)), []),
            nn.Linear(mid_dim, in_out_dim),
            nn.BatchNorm1d(in_out_dim),
        )
        self.scale = nn.Parameter(torch.zeros(1, in_out_dim))
        # configure mask
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        assert mask_config in {0, 1}
        self.mask = torch.zeros(in_out_dim, device=device) if not mask_config else torch.ones(in_out_dim, device=device)
        self.mask[::2] = 1 - mask_config

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        #x = x.to(self.device)
        if not reverse:
            x1, x2 = self.mask * x, (1. - self.mask) * x
            b1, b2 = self.s(x1), self.t(x1) * (1. - self.mask)
            y1, y2 = x1, x2 * torch.exp(b1) + b2
            return y1 + y2, log_det_J - b1.sum(dim=1)

        y1, y2 = self.mask * x, (1. - self.mask) * x
        b1, b2 = self.s(y1) * (1. - self.mask), self.t(y1) * (1. - self.mask)
        x1, x2 = y1, (y2 - b2) * torch.exp(-b1)
        x2[torch.isnan(x2)] = 0 # division by 0 of the mask is nan
        return x1 + x2, log_det_J

"""Log-scaling layer.
"""
class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        scale = torch.exp(self.scale) + self.eps
        det = torch.sum(self.scale)
        return x * (scale if not reverse else scale.reciprocal()), det


"""Standard logistic distribution.
"""
logistic = TransformedDistribution(Uniform(0, 1), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])

"""NICE main model.
"""
class NICE(nn.Module):
    def __init__(self, prior, coupling,
        in_out_dim, mid_dim, hidden, bottleneck, compress, device, n_layers):
        """Initialize a NICE.

        Args:
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device
        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        elif prior == 'logistic':
            self.prior = TransformedDistribution(Uniform(torch.tensor(0.).to(device), torch.tensor(1.).to(device)), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])
        else:
            raise ValueError('Prior not implemented.')

        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.n_layers = n_layers
        layer = AdditiveCoupling if coupling == 'additive' else AffineCoupling
        self.coupling_layers = nn.ModuleList(
            [layer(in_out_dim, mid_dim, hidden, i % 2) for i in range(self.n_layers)]
        ).to(device)
        self.scale = Scaling(in_out_dim).to(device)
        self.bottleneck_factor = compress
        self.bottleneck_loss = nn.MSELoss()
        self.bottleneck = bottleneck


    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, det = self.scale(z, reverse=True)

        for layer in reversed(self.coupling_layers):
            x, _ = layer(x, 0, reverse=True)


        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        log_det_J = 0
        for layer in self.coupling_layers:
            x, log_det_J = layer(x, log_det_J)
        z, det = self.scale(x)
        return z, log_det_J + det


    def loss(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)

        slices = [z[:, i::self.bottleneck_factor] for i in range(self.bottleneck_factor)]
        s = torch.stack(slices).permute(1, 0, 2)
        if self.bottleneck == 'redundancy':
            bottleneck_loss = torch.sum(torch.var(s, dim=1), dim=1)
            log_ll = 0.0
            for slice in slices:
                log_ll += torch.sum(self.prior.log_prob(slice), dim=1)
        if self.bottleneck == 'null':
            winner = slices[-1]
            loser = torch.ones_like(winner)
            bottleneck_loss = 0.0
            for slice in slices[:-1]:
                bottleneck_loss += self.bottleneck_loss(slice, loser)
            log_ll = torch.sum(self.prior.log_prob(winner), dim=1)
        log_det_J -= np.log(256)*self.in_out_dim #/ self.bottleneck_factor #log det for rescaling from [0.256] (after dequantization) to [0,1]

        #log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J, bottleneck_loss

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim // self.bottleneck_factor)).to(self.device)

        z_tag = torch.zeros((size, self.in_out_dim)).to(self.device)
        if self.bottleneck == 'redundancy':
            for i in range(self.bottleneck_factor):
                z_tag[:, i::self.bottleneck_factor] = z
        if self.bottleneck == 'null':
            for i in range(self.bottleneck_factor - 1):
                z_tag[:, i::self.bottleneck_factor] = torch.ones_like(z)
            z_tag[:, self.bottleneck_factor-1::self.bottleneck_factor] = z
        return self.f_inverse(z_tag)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        x = x.to(self.device)
        return self.loss(x)
