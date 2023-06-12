import numpy as np


def compute_non_standardness(ev):
    return (ev.sum(-1) - ev.shape[-1] - np.log(ev).sum(-1)) / 2


def compute_non_standardness_centered(ev):
    return -np.log(ev).sum(-1) / 2
