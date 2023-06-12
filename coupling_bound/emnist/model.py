from functools import wraps, partial
from typing import Iterable, Tuple

import torch
from FrEIA.framework import SequenceINN
import FrEIA.modules as Fm
from scipy.stats import ortho_group
from torch import Tensor, nn

from coupling_bound.emnist.permute import AxisFixedLinearTransform


def layer_requires_grad(layer):
    for param in layer.parameters():
        if param.requires_grad:
            return True
    return False


class IterativeINN(SequenceINN):
    def __init__(self, *dims: int, force_tuple_output=False):
        super().__init__(*dims, force_tuple_output=force_tuple_output)
        self.register_buffer("grad_layers", torch.tensor(-1))
        self.register_buffer("active_layers", torch.tensor(-1))

    def set_grad_layers(self, grad_layers):
        if not torch.is_tensor(grad_layers):
            grad_layers = torch.tensor(grad_layers)
        self.grad_layers = grad_layers

    def grad_parameters(self):
        for grad_layer in self.grad_layers:
            yield from self.module_list[grad_layer].parameters()

    def set_active_layers(self, active_layers):
        if not torch.is_tensor(active_layers):
            active_layers = torch.tensor(active_layers)
        self.active_layers = active_layers

    def forward(self, x_or_z: Tensor, c: Iterable[Tensor] = None,
                rev: bool = False, jac: bool = True) -> Tuple[Tensor, Tensor]:
        if len(self.grad_layers.shape) == 0:
            raise ValueError("Set grad_layers before using IterativeINN")
        if len(self.active_layers.shape) == 0:
            raise ValueError("Set active_layers before using IterativeINN")

        iterator = range(len(self.module_list))
        log_det_jac = 0

        if rev:
            iterator = reversed(iterator)

        if torch.is_tensor(x_or_z):
            x_or_z = (x_or_z,)

        grad_prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        for layer_idx in iterator:
            if layer_idx in self.active_layers:
                if layer_idx in self.grad_layers:
                    torch.set_grad_enabled(grad_prev)
                layer = self.module_list[layer_idx]
                if self.conditions[layer_idx] is None:
                    x_or_z, j = layer(x_or_z, jac=jac, rev=rev)
                else:
                    x_or_z, j = layer(x_or_z, c=[c[self.conditions[layer_idx]]],
                                      jac=jac, rev=rev)
                log_det_jac = j + log_det_jac

        torch.set_grad_enabled(grad_prev)
        return x_or_z if self.force_tuple_output else x_or_z[0], log_det_jac


def subnet_fc(c_in, c_out, width):
    subnet = nn.Sequential(nn.Linear(c_in, width), nn.ReLU(),
                           nn.Linear(width, width), nn.ReLU(),
                           nn.Linear(width, c_out))
    for l in subnet:
        if isinstance(l, nn.Linear):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet


def subnet_fc_of_width(width):
    return partial(subnet_fc, width=width)


def subnet_conv(c_in, c_out, width):
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet


def subnet_conv1(c_in, c_out):
    return subnet_conv(c_in, c_out, 16)


def subnet_conv2(c_in, c_out):
    return subnet_conv(c_in, c_out, 32)


def add_permute(inn, soft):
    if soft:
        n_channels = inn.shapes[-1][0]
        random_rot = torch.from_numpy(ortho_group.rvs(n_channels)).float()
        inn.append(AxisFixedLinearTransform, M=random_rot)
    else:
        inn.append(Fm.PermuteRandom)


def build_opt_append(inn, len_tot: int, max_depth: int, truncate_left: bool):
    if max_depth > -1:
        if truncate_left:
            counter = len_tot - 1
            counter_inc = -1
        else:
            counter = 0
            counter_inc = +1

        @wraps(inn.append)
        def opt_append(*args, **kwargs):
            nonlocal counter
            if counter < max_depth:
                inn.append(*args, **kwargs)
                # Assertion that only layers with parameters are added via
                # `opt_append`
                assert layer_requires_grad(inn[-1])
            counter += counter_inc
    else:
        opt_append = inn.append
    return opt_append


def coupling_architecture(all_in_one, soft_permute):
    if all_in_one:
        coupling_block = Fm.AllInOneBlock
        coupling_args = dict(permute_soft=soft_permute)
        coupling_permute = (lambda *args: None)
        coupling_factor = 2
    else:
        coupling_block = Fm.GLOWCouplingBlock
        coupling_args = dict()
        coupling_permute = add_permute
        coupling_factor = 1
    return coupling_block, coupling_args, coupling_permute, coupling_factor


def construct_net_moons(max_depth: int, soft_permute: bool,
                        truncate_left: bool, fc: bool, all_in_one: bool,
                        fc_width: int):
    assert fc, "Moons data set only supported with fc layers"
    inn = IterativeINN(2)

    cpl_block, cpl_args, cpl_permute, cpl_factor = coupling_architecture(
        all_in_one, soft_permute)

    len_tot = 4 * cpl_factor

    opt_append = build_opt_append(inn, len_tot, max_depth, truncate_left)

    for k in range(len_tot):
        opt_append(cpl_block,
                   subnet_constructor=subnet_fc_of_width(fc_width), **cpl_args)
        cpl_permute(inn, soft_permute)

    return inn


def construct_net_emnist(max_depth: int, soft_permute: bool,
                         truncate_left: bool, fc: bool, all_in_one: bool,
                         fc_width: int):
    inn = IterativeINN(1, 28, 28)

    cpl_block, cpl_args, cpl_permute, cpl_factor = coupling_architecture(
        all_in_one, soft_permute)

    len_conv1 = (0 if fc else 4) * cpl_factor
    len_conv2 = (0 if fc else 4) * cpl_factor
    len_fc = (10 if fc else 2) * cpl_factor
    len_tot = len_conv1 + len_conv2 + len_fc

    opt_append = build_opt_append(inn, len_tot, max_depth, truncate_left)
    inn.append(Fm.IRevNetDownsampling)

    for k in range(len_conv1):
        opt_append(cpl_block, subnet_constructor=subnet_conv1, **cpl_args)
        cpl_permute(inn, soft_permute)
    inn.append(Fm.IRevNetDownsampling)

    for k in range(len_conv2):
        opt_append(cpl_block, subnet_constructor=subnet_conv2, **cpl_args)
        cpl_permute(inn, soft_permute)
    inn.append(Fm.Flatten)

    for k in range(len_fc):
        opt_append(cpl_block,
                   subnet_constructor=subnet_fc_of_width(fc_width), **cpl_args)
        cpl_permute(inn, soft_permute)

    return inn


def construct_net(data_set, max_depth: int, soft_permute: bool,
                  truncate_left: bool, fc: bool, all_in_one: bool,
                  fc_width: int):
    if data_set == "EMNIST":
        return construct_net_emnist(max_depth, soft_permute, truncate_left, fc,
                                    all_in_one, fc_width)
    elif data_set == "moons":
        return construct_net_moons(max_depth, soft_permute, truncate_left, fc,
                                   all_in_one, fc_width)
    else:
        raise ValueError(f"Dataset {data_set!r} not known.")
