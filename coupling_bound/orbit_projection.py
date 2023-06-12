from collections import defaultdict
from itertools import product
from math import prod

import gmpy2
import mpmath
import numpy as np
import sympy as sp
from scipy.special import comb
from tqdm import tqdm

import coupling_bound as cb


class Vector:
    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return self.inner.shape[0]

    def __getitem__(self, entry):
        return self.inner[entry, 0]

    def __getattr__(self, key):
        return getattr(self.inner, key)


def M1(a, y, indicators=None):
    # todo make type safe
    n = y.shape[0]
    i = sp.Dummy("i")
    r, s = sp.Dummy("r"), sp.Dummy("s")
    the_prod = (
            sp.Product((y[i, 0] - y[r, 0]), (r, 0, i - 1)) *
            sp.Product((y[i, 0] - y[s, 0]), (s, i + 1, n - 1))
    )
    if indicators is not None:
        the_ind = indicators[i, 0]
    else:
        the_ind = cb.orthog_monomial.Indicator(y[i, 0] > a)
    the_sum = sp.Sum(the_ind * (y[i, 0] - a) ** (n - 2) / the_prod,
                     (i, 0, n - 1))
    return (n - 1) * the_sum


def compute_cNK(N, K, log=False):
    i = sp.Dummy("i")
    log_fn = sp.log if log else lambda arg: arg
    accumulator = sp.Sum if log else sp.Product
    cNK = accumulator(log_fn(sp.binomial(N - K + i, i)), (i, 1, K - 1))
    return cNK


def vandermonde_polynomial(x, analytic, log=False):
    if analytic:
        if isinstance(x, sp.MatrixSymbol):
            x = Vector(x)

        N = x.shape[0]
        log_fn = sp.log if log else lambda arg: arg
        return (sp.Add if log else sp.Mul)(*[
            log_fn(x[j] - x[i])
            for i in range(N)
            for j in range(i + 1, N)
        ])
    else:
        N = x.shape[-1]
        log_fn = np.log if log else lambda arg: arg
        return (np.sum if log else np.prod)(
            np.stack([
                log_fn(x[..., j] - x[..., i])
                for i in range(N)
                for j in range(i + 1, N)
            ], axis=0)
            , axis=0
        )


def compute_Vx(K, x, analytic, log=False):
    if analytic:
        if isinstance(x, (sp.MatrixSymbol, sp.Matrix)):
            x = Vector(x)

        N = len(x)
        log_fn = sp.log if log else lambda arg: arg
        Vx = (sp.Add if log else sp.Mul)(*[
            log_fn(x[j] - x[i])
            for i in range(K)
            for j in range(N - K + 1 + i, N)
        ])
        return Vx
    else:
        N = x.shape[-1]
        if K == 1:
            return 0 if log else 1
        log_fn = np.log if log else lambda arg: arg
        return (np.sum if log else np.prod)(
            np.stack([
                log_fn(x[..., j] - x[..., i])
                for i in range(K)
                for j in range(N - K + 1 + i, N)
            ], axis=0)
            , axis=0
        )


def nu_N_K(N, K):
    """
    Follow notation from Olshanski, 2013.

    :param N:
    :param K:
    :param x:
    :return:
    """

    a = sp.MatrixSymbol("a", K, 1)
    x = sp.MatrixSymbol("x", N, 1)

    indicators = sp.MatrixSymbol("I", K, N)

    Va = vandermonde_polynomial(a)
    cNK = compute_cNK(N, K)
    Vx = compute_Vx(K, x, True)
    Mmat = sp.ImmutableMatrix([
        [
            M1(a[j, 0], x[i:N - K + i + 1, 0], indicators[j, i:N - K + i + 1])
            for j in range(K)
        ]
        for i in range(K)
    ])

    return cNK * Va * sp.det(Mmat) / Vx, a, x, indicators


def face_indicators(indicators, *ls):
    indicator_subs = {}
    for i in range(indicators.shape[0]):
        for j in range(indicators.shape[1]):
            # Is a_i < x_j?
            indicator_subs[indicators[i, j]] = 1 if ls[i] < j else 0
    return indicator_subs


def identify_integrals(x, a):
    if isinstance(x, sp.MatrixSymbol):
        x = list(x)
    if isinstance(a, sp.MatrixSymbol):
        a = list(a)

    N = len(x)
    K = len(a)

    interlacing_ranges = [
        tuple(range(i, i + N - K))
        for i in range(K)
    ]

    for start_points in product(*interlacing_ranges):
        if any(x1 > x2 for x1, x2 in zip(start_points[:-1], start_points[1:])):
            continue

        unique_start_points = set(start_points)
        a_limits = {}
        for usp in sorted(unique_start_points):
            ais = []
            for ia, start_point in zip(range(K), start_points):
                if start_point == usp:
                    ais.append(a[ia])

            for ai, left, right in zip(ais,
                                       [x[usp]] + ais[:-1],
                                       ais[1:] + [x[usp + 1]]):
                a_limits[ai] = left, right

        yield (
            tuple(a_limits[a[ia]] for ia in range(K)),
            start_points
        )


def marginal_densities(N, K):
    density, a, x, indicators = nu_N_K(N, K)
    density = density.doit()
    simple_density = sp.simplify(density)

    x_ = sp.symbols(f"x1:{N + 1}")
    a_ = sp.symbols(f"a1:{K + 1}")

    density_ = simple_density.subs(dict(zip(x, x_))).subs(dict(zip(a, a_)))

    marginals = []
    for ia in tqdm(range(K)):
        parts = defaultdict(int)
        for limits, left_idx in tqdm(list(identify_integrals(x_, a_))):
            # Marginalise variable by variable in this polynomial part of
            # the density
            indicator_subs = face_indicators(indicators, *left_idx)
            part_density = sp.simplify(density_.subs(indicator_subs))
            for ja in range(K):
                if ia == ja:
                    continue
                # Adapt integration bounds to account for integrations
                # already performed
                left, right = limits[ja]
                if left in a_:
                    if left_idx[ia] != left_idx[ja] or ja < ia:
                        left = left_idx[ja]
                    else:
                        left = a_[ia]
                part_density = sp.integrate(part_density, (a_[ja], left, right))
            parts[left_idx[ia]] += part_density
        marginals.append({
            key: sp.simplify(value)
            for key, value
            in parts.items()
        })

    return marginals


def piecewise_from_marginal(ai, x, parts):
    if isinstance(x, sp.MatrixSymbol):
        x = list(x)

    return sp.Piecewise(
        *[
            (part_density, (x[ileft] < ai) & (ai < x[ileft + 1]))
            for ileft, part_density
            in parts.items()
        ],
        (0, True)
    )


def divided_differences(fn, knots):
    # Not directly applied to array so that fn can decide which library to use
    # Swap axes such that dimensions are first axis for easy iteration
    prev = [fn(knot) for knot in np.swapaxes(knots, 0, -1)]

    n_knots = knots.shape[-1]
    for j in range(1, n_knots):
        prev = [
            (prev[i + 1] - prev[i]) / (knots[..., i + j] - knots[..., i])
            for i in range(n_knots - j)
        ]
    assert len(prev) == 1, prev

    return prev[0]


def peano_reciprocal(n_knots):
    def fn(var):
        lib = sp if isinstance(var, sp.Basic) else np
        Z = max(1, n_knots - 1)
        return var ** (n_knots - 2) * lib.log(var) * Z

    return fn


def peano_monomial(n_knots, p):
    def fn(var):
        lib = sp if isinstance(var, sp.Basic) else np
        if lib is sp:
            Z = sp.binomial(p + n_knots - 1, p) ** -1
        else:
            Z = comb(p + n_knots - 1, p) ** -1
        return var ** (p + n_knots - 1) * Z

    return fn


def power_moment(knots, p):
    n_knots = knots.shape[-1]
    if p == -1:
        fn = peano_reciprocal(n_knots)
    else:
        fn = peano_monomial(n_knots, p)

    return divided_differences(fn, knots)


def moment_matrix(knots, K):
    if not isinstance(knots, np.ndarray):
        knots = np.array(knots)
    N = knots.shape[-1]
    n_knots = N - K + 1

    M = np.zeros((*knots.shape[:-1], K, K), dtype=knots.dtype)
    for j, p in enumerate([-1] + list(range(1, K))):
        for i in range(K):
            sub_knots = knots[..., i:i + n_knots]
            M[..., i, j] = power_moment(sub_knots, p)
    return M


MODE_MOMENT_MATRIX = "moment_matrix"
MODE_MODIFIED_VANDERMONDE = "modified_vandermonde"
MODE_DIRECT_DETERMINANT = "direct_determinant"


def flexible_log_function(dtype):
    if dtype is sp.Rational:
        def log_fn(arg):
            return np.vectorize(sp.Rational)(np.log(arg))
    elif dtype is mpmath.mpf:
        log_fn = np.vectorize(mpmath.log)
    elif dtype is gmpy2.mpfr:
        log_fn = np.vectorize(gmpy2.log)
    elif dtype is sp.Basic:
        log_fn = np.vectorize(sp.log)
    else:
        log_fn = np.log

    return log_fn


def reciprocal_sum_log_expectation(eigenvalues, K, mode, dtype=None):
    N = eigenvalues.shape[-1]
    if K == N:
        return flexible_log_function(dtype)(np.sum(1 / eigenvalues, -1))
    if K == 0:
        return 0

    log_fn = flexible_log_function(dtype)
    if mode == MODE_MOMENT_MATRIX:
        analytic = dtype is sp.Basic
        mom_mat = moment_matrix(eigenvalues, K)
        if analytic:
            mom_mat_log_det = sp.log(sp.det(sp.Matrix(mom_mat)))
        else:
            mom_mat_log_det = np.linalg.slogdet(mom_mat)[1]
        cNK = compute_cNK(N, K, log=True).doit()
        if not analytic:
            cNK = float(cNK)

        constants = (
                cNK - compute_Vx(K, eigenvalues, analytic=analytic, log=True)
        )
        return mom_mat_log_det + constants
    elif mode == MODE_MODIFIED_VANDERMONDE:
        analytic = dtype is sp.Basic
        constants = (
                log_fn(N - K)
                - vandermonde_polynomial(eigenvalues, analytic=analytic, log=True)
        )
        M = modified_vandermonde_matrix(eigenvalues, K, log_fn)
        if analytic:
            det_M = sp.log(sp.det(sp.Matrix(M)))
        else:
            det_M = np.linalg.slogdet(M)[1]
        return constants + det_M
    elif mode == MODE_DIRECT_DETERMINANT:
        terms = direct_determinant_terms(K, eigenvalues, log_fn)
        return log_fn(N - K) + log_fn(np.sum(terms, -1))
    else:
        raise ValueError(f"Mode {mode} is not known.")


def reciprocal_sum_expectation(eigenvalues, K, dtype=None, other_sums=None):
    N = eigenvalues.shape[-1]
    if K == N:
        return np.sum(1 / eigenvalues, -1)
    log_fn = flexible_log_function(dtype)
    terms = direct_determinant_terms(eigenvalues, K, log_fn,
                                     other_sums=other_sums)
    return (N - K) * np.sum(terms, -1)


def direct_determinant_terms(eigenvalues, K, log_fn, other_sums=None):
    if not isinstance(eigenvalues, np.ndarray):
        eigenvalues = np.array(eigenvalues)
    N = eigenvalues.shape[-1]

    # R values
    v_rests = diff_apply(eigenvalues, np.prod)
    # S values
    if other_sums is None:
        _, other_sums = sum_prod_comb_rest(eigenvalues, K - 1)
    # Compute terms
    return (
            (-1) ** (N - K)
            * log_fn(eigenvalues)
            * eigenvalues ** (N - K - 1)
            / v_rests
            * other_sums
    )


def diff_apply(eigenvalues, method=np.prod, neutral_element=1):
    N = eigenvalues.shape[-1]
    diff_mat = eigenvalues[..., None, :] - eigenvalues[..., :, None]
    diag = np.arange(N)
    diff_mat[..., diag, diag] = diff_mat[..., diag, diag] + neutral_element
    v_rests = method(diff_mat, -1)
    return v_rests


def sum_prod_comb_rest(values: np.ndarray, K, collect=False):
    """
    Computes
    - the sum of all length K combinations of values,
    - the sum of all length K combinations of values except each value.

    Inspired by https://math.stackexchange.com/a/1370175/869094

    :param values:
    :param K:
    :param collect:
    :return:
    """
    # Way to get arrays of shape and dtype of values with entry 1
    sum_k_wo_values = values * 0 + 1
    sum_k = sum_k_wo_values[..., :1]

    if collect:
        sum_ks_wo_values = np.tile(
            sum_k_wo_values, [K + 1, *[1] * len(values.shape)]
        )
        sum_ks_wo_values[0] = sum_k_wo_values
    else:
        sum_ks_wo_values = None
    for k in range(1, K + 1):
        sum_k = (values * sum_k_wo_values).sum(-1, keepdims=True) / k
        # sum_k = sum_k.astype(values.dtype)
        sum_k_wo_values = sum_k - values * sum_k_wo_values
        if collect:
            sum_ks_wo_values[k] = sum_k_wo_values
    return sum_k[..., 0], sum_ks_wo_values if collect else sum_k_wo_values


def modified_vandermonde_matrix(eigenvalues, K, log_fn):
    if not isinstance(eigenvalues, np.ndarray):
        eigenvalues = np.array(eigenvalues)

    N = eigenvalues.shape[-1]
    M = np.zeros((*eigenvalues.shape[:-1], N, N), dtype=eigenvalues.dtype)
    for i in range(N):
        for j in range(N):
            aj = eigenvalues[..., j]
            if i == N - K:
                M[..., i, j] = aj ** (N - K - 1) * log_fn(aj)
            else:
                M[..., i, j] = aj ** i
    return M


def other_form_terms(eigenvalues, K, dtype=None):
    contributions = 0
    N = eigenvalues.shape[-1]
    elementary_polys = sum_prod_comb_rest(eigenvalues, K - 1, True)[1]
    for k in range(K):
        elementary_sum = 0
        for i in range(k + 1):
            ev_prod = prod((eigenvalues[i] - eigenvalues[j]) for j in range(k + 1, ))
            elementary_sum += (-1) ** (N + i + 1) * ev_prod * elementary_polys[i]
        amplitude = eigenvalues[..., k + N - K] - eigenvalues[..., k]
        contributions += amplitude * elementary_sum
    return contributions
