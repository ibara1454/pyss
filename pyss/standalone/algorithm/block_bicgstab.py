#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.sparse.linalg


def block_bicgstab(A, B, x0=None, tol=1e-10, maxiter=None, M=None):
    """
    Linear solver of Ax = B.
    Solver for general matrix A and B.

    Parameters
    ----------
    A : (n, n) sparse matrix
        Real or complex square matrix of shape (n, n).
    B : (n, m) sparse matrix
        Real or complex vector representing the right hand side of the
        equation.
    x0 : (n, m) sparse matrix | dense matrix
        Initial guess of the equation.
    tol : float
        Tolerance to achieve.
    maxiter : integer
        Maximum number of iterations.
    M : (n, n) sparse matrix | dense matrix | LinearOperator
        Preconditioner for A.

    Returns
    -------
    X : (n, m) ndarray of sparse matrix
        The solution of the sparse linear equation. If b is a vector,
        then x is also a vector of which size.
    info :
    """
    # TODO: (IMPORTANT!) use more effecient block linear solver,
    # consider blocked Bicgrq
    return scipy.sparse.linalg.spsolve(A, B)