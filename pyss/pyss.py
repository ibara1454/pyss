#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import scipy
import scipy.sparse.linalg
import operator as op
from . import util


def solve(A, B, L, M, N, center, radius):
    """
    Sakurai-Sugiura method, a paralleled generalized eigenpair solver.
    """
    S = build_complex_moment(A, B, L, M, N, center, radius)
    U, _, _ = util.trimmed_svd(S)
    # Solve the reduced eigenvalue problem
    eigvals, eigvecs = util.shifted_rayleigh_ritz(A, B, U, shift=center)
    return eigvals, eigvecs


def build_complex_moment(A, B, L, M, N, center, radius):
    """
    Build complex moment corresponding to the given number k.

    Parameters
    ----------
    k : int
        Degree of complex moment. 0 <= k < M

    Returns
    -------
    S : (n*L) array
    The k-th complex moment, of shape (rows, cols).
    """
    radian2cartesian = numpy.vectorize(lambda rad: center + radius * numpy.exp(rad * 1j))
    n, _ = A.shape
    V = build_source_matrix(rows=n, cols=L)
    BV = B.dot(V)
    zs = radian2cartesian(numpy.arange(N) * 2 * numpy.pi / N)
    # Calculate all f(z) values on contour
    # TODO: Culculate f(z) parallelly
    # TODO: Use more effective linear solver
    Ys = list(map(lambda z: scipy.sparse.linalg.spsolve(z * B - A, BV), zs))
    # Build each k's complex moment
    # TODO: build complex moments parallelly
    h = 2 * numpy.pi / N
    def moment_k(k):
        return util.contour_integral(list(map(op.mul, Ys, (zs - center) * zs ** k)), h, samplep=3) / (2 * numpy.pi)
    S_ks = map(moment_k, range(M))
    # Combine S_k's horizontally
    return numpy.hstack(mat for mat in S_ks)


def build_source_matrix(rows, cols):
    """
    Build a source matrix with size (rows, cols).

    Parameters
    ----------
    rows : int
        Size of rows.
    cols : int
        Size of columns.

    Returns
    -------
    V : (rows, cols) array
        Source matrix, of shape (rows, cols).
    """
    return numpy.random.rand(rows, cols)
