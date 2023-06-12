from math import log

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


def rand_exp(min_ev, max_ev, batch_size, dim):
    min_l, max_l = log(min_ev), log(max_ev)

    samples = torch.rand((batch_size, dim), dtype=torch.float64)
    samples = samples * (max_l - min_l) + min_l
    samples = torch.exp(samples)

    return samples


def non_degenerate_ones(shape, epsilon=1e-5):
    try:
        shape = tuple(shape)
    except TypeError:
        shape = shape,
    return torch.tile(
        torch.linspace(1 - epsilon, 1 + epsilon, shape[-1]), (*shape[:-1], 1)
    )


def build_dataset(dim, kind_size=1024, min_ev=1e-16, max_ev=1e3,
                  degenerate=True, centered=False):
    # Don't consider the case where the special eigenvalue is 1
    evs_big = torch.from_numpy(np.geomspace(1, max_ev, kind_size + 1)[1:])
    evs_small = torch.from_numpy(np.geomspace(min_ev, 1, kind_size + 1)[:-1])

    if degenerate:
        ones = torch.ones
    else:
        ones = non_degenerate_ones

    sources = {
        "Single Varying > 1": torch.cat([
            evs_big.reshape(kind_size, 1),
            ones((kind_size, dim - 1))
        ], 1),
        "Single Varying < 1": torch.cat([
            evs_small.reshape(kind_size, 1),
            ones((kind_size, dim - 1))
        ], 1),
        "All Varying But One > 1": torch.cat([
            evs_big.reshape(kind_size, 1) * ones((kind_size, dim - 1)),
            ones((kind_size, 1))
        ], 1),
        "All Varying But One < 1": torch.cat([
            evs_small.reshape(kind_size, 1) * ones((kind_size, dim - 1)),
            ones((kind_size, 1))
        ], 1),
        "All Varying But One (Shifted) > 1": torch.cat([
            evs_big.reshape(kind_size, 1),
            (dim - evs_big.reshape(kind_size, 1))
            / (dim - ones((1, dim - 1)))
        ], 1)[evs_big < dim],
        "Uniform [0, 5]": torch.rand(kind_size, dim, dtype=torch.float64) * 5,
        "Uniform [0, 2]": torch.rand(kind_size, dim, dtype=torch.float64) * 2,
        "Log Uniform": rand_exp(min_ev, max_ev, kind_size, dim),
        "Half Big, Half Small": torch.cat([
            evs_big.reshape(kind_size, 1) * ones((1, dim // 2)),
            evs_small.reshape(kind_size, 1) * ones((1, dim - dim // 2)),
        ], 1),
        "Degenerate > 1": evs_big.reshape(kind_size, 1) * ones((1, dim)),
        "Degenerate < 1": evs_small.reshape(kind_size, 1) * ones((1, dim))
    }
    if centered:
        del sources["Degenerate > 1"]
        del sources["Degenerate < 1"]

        new_sources = {}
        for label, data in sources.items():
            # centered = data - torch.minimum(
            #    data.min(-1, keepdim=True).values - min_ev,
            #    data.mean(-1, keepdim=True) + 1
            # )
            standardized = data / data.mean(-1, keepdim=True)
            new_sources[label] = standardized
        sources = new_sources

    return sources


def data_normalized_by_channel(dataset_name, noise, root):
    if dataset_name == "MNIST":
        dataset = torchvision.datasets.MNIST(root, train=True)
    elif dataset_name == "EMNIST":
        dataset = torchvision.datasets.EMNIST(root, split='digits', train=True)
    elif isinstance(dataset_name, Dataset):
        dataset = dataset_name
    else:
        raise ValueError(f"Dataset {dataset_name!r} is not known")

    if torch.is_tensor(dataset.data):
        data = dataset.data
    else:
        data = torch.from_numpy(dataset.data)
    if len(data.shape) == 3:
        data = data.unsqueeze(-1)
    ds_float = data.float().reshape(len(dataset), -1, data.shape[-1])
    ds_means = ds_float.reshape(-1, data.shape[-1]).mean(0, keepdim=True)
    ds_centered = ds_float - ds_means
    ds_stds = ds_centered.reshape(-1, data.shape[-1]).std(0, unbiased=False)
    ds_normalized = ds_centered / ds_stds

    ds_sub = ds_normalized
    ds_noisy = (ds_sub + torch.randn_like(ds_sub) * noise).reshape(
        len(ds_sub), -1)
    ds_stds = ds_noisy.reshape(-1, data.shape[-1]).std(0, unbiased=False)
    ds_noisy_normalized = ds_noisy / ds_stds
    return ds_noisy_normalized


def dataset_cov(dataset_name: str, noise: float, root: str):
    ds_noisy_normalized = data_normalized_by_channel(dataset_name,
                                                     noise, root)
    return ds_noisy_normalized.T @ ds_noisy_normalized / len(
        ds_noisy_normalized)
