from itertools import chain

import mpmath
import numpy as np
from tqdm.auto import tqdm

from coupling_bound import orbit_projection
from coupling_bound.loss import compute_non_standardness_centered


def compute_var_max_bound(ev, dim_a):
    dim = ev.shape[-1]
    dim_p = dim - dim_a

    loss = compute_non_standardness_centered(ev)
    var = np.var(ev, -1)
    ev_max = np.max(ev, -1)
    ev_mean = ev.mean(-1)
    inner_bound = 1 - dim * dim_p / ((dim + 2) * (dim - 1)) * var / ev_max
    return loss + (
            dim_a * np.log(inner_bound)
            + dim * np.log(ev_mean)
    ) / 2


def compute_unitary_bound(ev, dim_a, batch_size=128, pbar=None, prec=1000):
    loss = compute_non_standardness_centered(ev)
    if dim_a == 0:
        return loss

    old_prec = mpmath.mp.prec
    try:
        mpmath.mp.prec = prec
        dim = ev.shape[-1]
        ev_mean = ev.mean(-1)

        ev_view = (ev / ev_mean[..., None]).reshape(-1, dim)
        expectations = np.zeros(ev_view.shape[0])
        batch_iter = range(0, expectations.shape[0], batch_size)
        if pbar is None:
            pbar = len(batch_iter) > 1
        for offset in tqdm(batch_iter, disable=not pbar):
            batch_ev = ev_view[offset:offset + batch_size]
            batch_out = float("nan")
            for noise in chain([0], np.geomspace(1e-15, 1e-5, 20)):
                try:
                    noisy_batch = batch_ev + np.random.randn() * noise
                    batch_out = orbit_projection.reciprocal_sum_expectation(
                        np.vectorize(mpmath.mpf)(1 / noisy_batch),
                        dim_a, dtype=mpmath.mpf
                    )
                    batch_out[np.abs(batch_out) != batch_out] = float("nan")
                    break
                except ZeroDivisionError:
                    pass
            expectations[offset:offset + batch_size] = batch_out

        return (
                loss
                + (
                        dim_a * np.log(expectations.reshape(ev.shape[:-1]))
                        - dim_a * np.log(dim_a)
                        + dim * np.log(ev_mean)
                ) / 2)
    finally:
        mpmath.mp.prec = old_prec


def compute_loss_only_bound_by_loss(loss, dim_a, dim):
    dim_p = dim - dim_a
    geom = np.exp(-2 * loss / dim)
    return loss + dim_a / 2 * np.log(
        1 - 2 * dim * dim_p / ((dim - 1) * (dim + 2))
        * (1 - np.sqrt(1 - geom ** dim)) / (1 + np.sqrt(1 - geom ** dim)) * (
                1 - geom)
    )


def compute_loss_only_bound(ev, dim_a):
    dim = ev.shape[-1]
    loss = compute_non_standardness_centered(ev)
    ev_mean = ev.mean(-1)
    bound_by_loss = compute_loss_only_bound_by_loss(loss, dim_a, dim)
    return bound_by_loss + dim * np.log(ev_mean) / 2
