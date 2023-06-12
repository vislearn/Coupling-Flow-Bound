import os.path
from functools import wraps
from math import ceil
from typing import Union, Sized

import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_by_kind(sources, x, out, *args, fmt=".", **kwargs):
    offset = 0
    for i, (label, data) in enumerate(sources.items()):
        selector = slice(offset, offset + len(data))
        plt.plot(x[selector], out[selector], fmt, label=label, *args, **kwargs)
        offset += len(data)


def plot_ratio_by_condition_number(sources, condition_number, ratio, logx=True,
                                   ylim=(0, 1), legend=True,
                                   xlabel=r"Condition number $\kappa$",
                                   ylabel=None, *args, **kwargs):
    plot_by_kind(sources, condition_number, ratio, *args, **kwargs)
    if logx:
        plt.xscale("log")
    if ylim is not None:
        plt.ylim(ylim)
    if legend:
        plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.xlabel(ylabel)
    rel_nan = torch.isnan(ratio).sum() / ratio.shape.numel()
    print(f"{rel_nan * 100:.0f}% of data is NaN.")


def subplot_grid(count, ncols=3, expand=True, width_each=None, height_each=None, figsize=None, **kwargs):
    if ncols is None:
        ncols = count
    if count < ncols:
        ncols = count
    nrows = ceil(count / ncols)

    if figsize is None:
        if width_each is None:
            width_each = 4
        if height_each is None:
            height_each = 4
        kwargs["figsize"] = (ncols * width_each, nrows * height_each)
    else:
        assert width_each is None, "Cannot give both figsize and width"
        assert height_each is None, "Cannot give both figsize and height"
        kwargs["figsize"] = figsize

    if nrows * ncols == 0:
        return None, None
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, **kwargs)
    if count % ncols != 0:
        for which in range(count % ncols - ncols, 0):
            axes[-1][which].axis("off")
    return fig, axes.reshape(-1)[:count] if expand else axes


@wraps(subplot_grid)
def iter_ax_grid(sized: Sized, *args, **kwargs):
    fig, axes = subplot_grid(len(sized), *args, **kwargs)
    for item, ax in zip(sized, axes):
        plt.sca(ax)
        yield item


def get_notebook_name() -> Union[str, None]:
    from jupyter_server import serverapp as app
    import ipykernel, requests, os
    kernel_id = os.path.basename(
        ipykernel.get_connection_file()
    ).split('-', 1)[1].split('.')[0]
    srv = next(app.list_running_servers())

    sessions = requests.get(
        srv['url'] + 'api/sessions?token=' + srv['token']).json()
    for session in sessions:
        if session["kernel"]["id"] == kernel_id:
            return srv["root_dir"] + "/" + session['notebook']['path']


@wraps(plt.savefig)
def savefig(*args, **kwargs):
    if "metadata" not in kwargs:
        nb_name = get_notebook_name()
        if nb_name is not None:
            kwargs["metadata"] = {"Producer": os.path.basename(nb_name)}
    return plt.savefig(*args, **kwargs)


def sympy_plot(var_spec, term, *args, geometric=False, npoints=200, subs=None,
               ax=None,
               **kwargs):
    import sympy as sp
    var, low, high = var_spec
    var_vals = (np.geomspace if geometric else np.linspace)(low, high, npoints)
    if subs is not None:
        term = term.subs(subs)
    term_fn = sp.lambdify(var, term)
    if ax is None:
        ax = plt.gca()
    return ax.plot(var_vals, term_fn(var_vals), *args, **kwargs)
