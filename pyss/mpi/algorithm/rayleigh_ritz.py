#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy
import scipy.linalg
import pyss.mpi.util.operation.hv as hv


def rayleigh_ritz(cv, p, a, b, comm, left=False, right=True,
                  check_finite=True, homogeneous_eigvals=False):
    """
    Rayleigh-Ritz Procedure using MPI parallelism.

    Parameters
    ----------
    cv : Callable object with type
        `cv(c: (M, M) array_like, v: ndarray, scomm: MPI_Comm) -> ndarray`.
        The function of multiplying of matrix `c` and matrix `v` on `scomm`,
        where `v` is ndarray with shape (M / rank, l) on each node of `scomm`.
        The return value of `cv` should be a ndarray, and with the same shape
        of `v`.
    p : (M, n) array_like
        Semi-unitary matrix.
    a : (M, M) array_like
        A complex or real matrix whose eigenvalues and eigenvectors will be
        computed.
    b : (M, M) array_like, optional
        Right-hand side matrix in a generalized eigenvalue problem.
        Default is None, identity matrix is assumed.
    left : bool, optional
        Whether to calculate and return left eigenvectors. Default is False.
    right : bool, optional
        Whether to calculate and return right eigenvectors. Default is True.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    homogeneous_eigvals : bool, optional
        If True, return the eigenvalues in homogeneous coordinates.
        In this case w is a (2, M) array so that:
        `w[1,i] a vr[:,i] = w[0,i] b vr[:,i]`
        Default is False.

    Returns
    -------
    w : (M,) or (2, M) double or complex ndarray
        The eigenvalues, each repeated according to its multiplicity.
        The shape is (M,) unless `homogeneous_eigvals=True`.
    vl : (M, M) double or complex ndarray
        The normalized left eigenvector corresponding to the eigenvalue `w[i]`
        is the column `vl[:,i]`. Only returned if `left=True`.
    vr : (M, M) double or complex ndarray
        The normalized right eigenvector corresponding to the eigenvalue `w[i]`
        is the column `vr[:,i]`. Only returned if `right=True`.
    Raises:
    -------
    LinAlgError
        If eigenvalue computation does not converge.
    """
    # Calculate (P^H A P) explicitly
    l = hv(p.T.conj(), cv(a, p, comm), comm)
    # Calculate (P^H B P) explicitly
    m = hv(p.T.conj(), cv(b, p, comm), comm)
    w, v = scipy.linalg.eig(l, m, left, right,
                            overwrite_a=False, overwrite_b=False,
                            check_finite, homogeneous_eigvals)
    return None
