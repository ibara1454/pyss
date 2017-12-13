#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import scipy
import scipy.linalg


def svd(a, comm, iters=1, compute_uv=True, overwrite_a=False):
    """
    Singular value decomposition with separeted matrix on MPI parallelism.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to decompose. Which matrix is separeted over `comm` in
        colume-direction.
    comm : MPI_comm
        MPI communicator.
    iters : int
        Iteration numbers of downgrade procedure.
        use `iter=1` while the condition number of `a` is small,
        use `iter=2` or further while the conditino number is large.
        Over 3 times, the accumulation of round-off error might cause the
        result with a large amount of error.
    compute_uv : bool, optional
        Whether to compute also U and Vh in addition to s. Default is True.
    overwrite_a : bool, optional
        Whether to overwrite a; may improve performance. Default is False.

    Returns
    -------
    U : Unitary matrix having left singular vectors as columns.
        Of shape (M, M) or (M, K), depending on full_matrices.
        Which matrix is separeted over `comm` in colume-direction.
    s : The singular values, sorted in non-increasing order.
        Of shape (K,), with K = min(M, N).
    Vh : Unitary matrix having right singular vectors as rows.
        Of shape (N, N) or (K, N) depending on full_matrices.
    For compute_uv=False, only s is returned.
    """
    a, b = __dg_proc_n(a, iters, comm)
    U, s, Vh = scipy.linalg.svd(b)
    U = a @ U
    return U, s, Vh


def __dg_proc_n(a, n, comm):
    row, col = a.shape
    b = numpy.eye(col)
    for _ in range(n):
        a, b_n = __dg_proc(a, comm)
        b = b_n @ b
    return a, b


def __dg_proc(a, comm):
    """
    Special downgrade procedure. Which decreases the condition number of
    the multiplication `a* a`, where a is a matrix with large condition number.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix with large condition number. Which is separeted over `comm` in
        colume-direction.
    comm : MPI_comm
        MPI communicator.
    Returns
    -------
    a' : (M, N) array_like
        The downgrade matrix of given matrix `a`. With equation `a = a' b`
    b : (N, N) array_like
        The production of this downgrade procedure. With equation `a = a' b`
    Raises
    ------
    ValueError
        If input array is not square.
    ComplexWarning
        If a complex-valued array with nonzero imaginary parts on the diagonal
        is given and hermitian is set to True.
    """
    eps = numpy.finfo(numpy.float).eps
    # Calculate C* C, it must be positive definite
    # but has large condition number
    c = a.T.conj() @ a
    c = comm.allreduce(c)
    # Do the same ldlt composition on all nodes
    l, d, _ = scipy.linalg.ldl(c)  # d is diagonal "matrix"
    d = numpy.diag(d).copy()  # because numpy.diag returns the read-only view
    # Replace the invalid value with machine epsilon (Magic!!)
    d[d <= 0] = eps
    d12 = numpy.diag(d / 2)
    d12i = __inv_diag(d12)
    # TODO: use more effecient way to calculate d12 @ li instead of inv
    # linalg.inv takes more time but have higher accuracy than lin.solve
    b = d12 @ scipy.linalg.inv(l)
    # The alternative, by using the equation `L^t X^t = D^t`
    # b = scipy.linalg.solve(l.T, d12).T
    return a @ l @ d12i, b


def __inv_diag(a):
    """
    Return the inverse of diagonal matrix `a`. The diagonal must be large
    than 0.

    Parameters
    ----------
    a : (N, N) array_like
        Diagonal matrix.
    Returns
    -------
    ai : (N, N) array_like
        Inverse matrix of given diagonal matrix.
    """
    d = numpy.diag(a)
    ai = numpy.diag(1 / d)
    return ai
