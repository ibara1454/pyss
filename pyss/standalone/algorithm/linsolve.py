#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy
import scipy.sparse.linalg


def linsolve(A, B):
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

    Returns
    -------
    X : (n, m) ndarray of sparse matrix
        The solution of the sparse linear equation. If b is a vector,
        then x is also a vector of which size.

    """
    # (IMPORTANT!) use more effecient block linear solver,
    # like blocked Bicgrq
    return scipy.sparse.linalg.spsolve(A, B)
