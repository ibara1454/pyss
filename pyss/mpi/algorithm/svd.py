#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import scipy
import scipy.linalg


def svd(a, comm, full_matrices=True, compute_uv=True, overwrite_a=False):
    """
    Singular value decomposition with separeted matrix on MPI parallelism.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to decompose. Which matrix is separeted over comm in
        colume-direction.
    comm : MPI_comm
        MPI communicator.
    full_matrices : bool, optional
        If True (default), U and Vh are of shape (M, M), (N, N). If False,
        the shapes are (M, K) and (K, N), where K = min(M, N).
    compute_uv : bool, optional
        Whether to compute also U and Vh in addition to s. Default is True.
    overwrite_a : bool, optional
        Whether to overwrite a; may improve performance. Default is False.

    Returns
    -------
    U : Unitary matrix having left singular vectors as columns.
        Of shape (M, M) or (M, K), depending on full_matrices.
    s : The singular values, sorted in non-increasing order.
        Of shape (K,), with K = min(M, N).
    Vh : Unitary matrix having right singular vectors as rows.
        Of shape (N, N) or (K, N) depending on full_matrices.
    For compute_uv=False, only s is returned.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.
    """
    a, b = __dg_decomp(a, comm)
    U, s, Vh = scipy.linalg.svd(b)
    return a @ U, s, Vh


def __dg_decomp(a, comm):
    """
    Special downgrade decomposition. Which decreases the condition number of
    the multiplication `a* a`, where a is a matrix with large condition number.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix with large condition number. Which is separeted over comm in
        colume-direction.
    comm : MPI_comm
        MPI communicator.
    Returns
    -------

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
    d = numpy.diag(d).copy()  # cause numpy.diag returns the read-only view
    # Replace the invalid value with machine epsilon (Magic!!)
    d[d <= 0] = eps
    d12 = numpy.diag(d / 2)
    d12i = __inv_diag(d12)
    lt = l.T
    return l @ d12i, d12 @ lt


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
