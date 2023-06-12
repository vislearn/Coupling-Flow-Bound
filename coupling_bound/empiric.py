import numpy as np
import torch
from scipy.stats import ortho_group, unitary_group
from tqdm.auto import tqdm

MODE_O = "ortho"
MODE_U = "unitary"


def sample_rotated_cov(cov_in, rots=128, mode=MODE_O):
    """
    Adds an axis -2 to cov_in with rotated versions of cov_in.
    The rotations are shared over all instances.

    :param cov_in:
    :param rots:
    :return:
    """
    dim = cov_in.shape[-1]
    assert dim == cov_in.shape[
        -2], f"Dimension mismatch: {dim} != {cov_in.shape[-2]}"
    rot, rot_T = sample_rot(dim, rots, mode)
    # Share the rotations over the batch dimensions
    for _ in range(len(cov_in.shape) - 2):
        rot = rot.unsqueeze(0)

    cov_rot = rot_T @ cov_in.unsqueeze(-3).to(rot) @ rot
    return cov_rot


def rotate_cov_randomly(cov_in, mode=MODE_O):
    """
    Rotates each item in cov_in randomly.
    """
    dim = cov_in.shape[-1]
    assert dim == cov_in.shape[
        -2], f"Dimension mismatch: {dim} != {cov_in.shape[-2]}"
    cov_in_reshaped = cov_in.reshape(-1, dim, dim)
    rot_count = cov_in_reshaped.shape[0]
    rot, rot_T = sample_rot(dim, rot_count, mode)

    cov_rot = rot_T @ cov_in_reshaped.to(rot) @ rot
    return cov_rot.reshape(cov_in.shape)


def sample_rot(dim, rot_count, mode):
    if mode == MODE_O:
        rot = ortho_group.rvs(dim, rot_count)
        rot_T = np.swapaxes(rot, -1, -2)
    elif mode == MODE_U:
        rot = unitary_group.rvs(dim, rot_count)
        rot_T = np.swapaxes(rot, -1, -2).conj()
    else:
        raise ValueError(f"Q sampling mode {mode} unknown")
    rot = torch.from_numpy(rot)
    rot_T = torch.from_numpy(rot_T)
    return rot, rot_T


def get_layer(dim_a, cov_in, s=True, t=True, sp=False):
    dim = cov_in.shape[-1]
    assert dim == cov_in.shape[
        -2], f"Dimension mismatch: {dim} != {cov_in.shape[-2]}"
    dim_p = dim - dim_a

    cov_aa, cov_ap, cov_pa, cov_pp = extract_blocks(cov_in, dim_a)
    cov_pp_I = cov_pp.inverse()
    cov_aa_pp = cov_aa - cov_ap @ cov_pp_I @ cov_pa

    if sp:
        r = torch.diag(cov_pp) ** (-1 / 2)
    else:
        r = torch.ones(dim_p).to(cov_in)

    if t:
        if s:
            s = torch.diag(cov_aa_pp) ** (-1 / 2)
        else:
            s = torch.ones(dim_a).to(cov_in)
        t = -torch.diag(s) @ cov_ap @ cov_pp_I
    else:
        if s:
            s = torch.diag(cov_aa) ** (-1 / 2)
        else:
            s = torch.ones(dim_a).to(cov_in)
        t = torch.zeros(dim_a, dim_p)

    return torch.cat([
        torch.cat([torch.diag(r), torch.zeros(dim_p, dim_a)], 1),
        torch.cat([t, torch.diag(s)], 1)
    ], 0)


def apply_layer(dim_a, cov_in, s=True, t=True, sp=False):
    dim = cov_in.shape[-1]
    assert dim == cov_in.shape[
        -2], f"Dimension mismatch: {dim} != {cov_in.shape[-2]}"
    dim_p = dim - dim_a

    # Removes correlation between active and passive dimension
    if t:
        cov_aa, cov_ap, cov_pa, cov_pp = extract_blocks(cov_in, dim_a)
        cov_bb = apply_t(cov_aa, cov_ap, cov_pa, cov_pp)
        cov_new = torch.cat([
            torch.cat([cov_pp, torch.zeros(*cov_in.shape[:-2], dim_p, dim_a)], -1),
            torch.cat([torch.zeros(*cov_in.shape[:-2], dim_a, dim_p), cov_bb], -1)
        ], -2)
    else:
        cov_new = cov_in

    # Rescales diagonal
    if s or sp:
        s_full = compute_s(cov_new)
        indices = torch.arange(dim)
        if not s:
            indices_p = indices[dim_p:]
            s_full[..., indices_p, indices_p] = 1
        if not sp:
            indices_a = indices[:dim_p]
            s_full[..., indices_a, indices_a] = 1
        cov_out = apply_s(s_full, cov_new)
    else:
        cov_out = cov_new

    return cov_out


def compute_loss_empirically(ev, dim_a, rot_count, rot_mode, pbar=False):
    dim = ev.shape[-1]
    rot, rot_I = sample_rot(dim, rot_count, rot_mode)
    if pbar:
        print("Have rotations")
    cov_diag = torch.diag_embed(torch.from_numpy(ev))[:, None]
    if rot_mode == MODE_U:
        cov_diag = cov_diag + 0j
    cov = rot[None] @ cov_diag @ rot_I[None]
    cov_out = torch.zeros_like(cov)
    batch_size = 64
    for offset in tqdm(range(0, cov.shape[0], batch_size), disable=not pbar):
        batch_out = apply_layer(dim_a, cov[offset:offset + batch_size],
                                        s=True, t=True, sp=True)
        cov_out[offset:offset + batch_size] = batch_out
    if pbar:
        print("Applied layer")
    return -np.linalg.slogdet(cov_out.numpy())[1] / 2


def extract_blocks(cov_in, dim_a):
    dim_p = cov_in.shape[-1] - dim_a
    cov_pp = cov_in[..., :dim_p, :dim_p]
    cov_pa = cov_in[..., :dim_p, dim_p:]
    cov_ap = cov_in[..., dim_p:, :dim_p]
    cov_aa = cov_in[..., dim_p:, dim_p:]
    return cov_aa, cov_ap, cov_pa, cov_pp


def apply_t(cov_aa, cov_ap, cov_pa, cov_pp):
    cov_pp_I = cov_pp.inverse()
    cov_bb = cov_aa - cov_ap @ cov_pp_I @ cov_pa
    return cov_bb


def apply_s(S, cov_bb):
    cov_cc = S.conj() @ cov_bb @ S
    return cov_cc


def compute_s(cov_bb):
    S = torch.diag_embed(torch.diagonal(cov_bb, dim1=-2, dim2=-1) ** -(1 / 2))
    return S


def neumann_inv(matrix, order, inv_init):
    identity = torch.eye(matrix.shape[-1]).to(matrix)
    for _ in range(len(matrix.shape) - 2):
        identity = identity.unsqueeze(0)

    approx = torch.zeros_like(matrix)
    running = identity
    for i in range(order + 1):
        approx = approx + running
        if i < order:
            running = running @ (identity - inv_init @ matrix)
    return approx @ inv_init


def newton_schulz_inv(matrix, k, inv_init):
    identity = torch.eye(matrix.shape[-1]).to(matrix)
    for _ in range(len(matrix.shape) - 2):
        identity = identity.unsqueeze(0)

    estimate = inv_init
    for _ in range(k):
        estimate = 2 * estimate - estimate @ matrix @ estimate
    return estimate
