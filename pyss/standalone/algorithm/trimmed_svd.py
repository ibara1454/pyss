#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import scipy.linalg
import warnings


def trimmed_svd(A, tol=1e-12, singular_value_lower_bound=1e-6):
    """
    Singular value decomposition, with trimming too small singular values.

    Parameters
    ----------
    A : (n, n) array_like
        A real or complex matrix of shape (n, n).
    tol : float, optional
        The (relative) tolerance.
        This method will trim the the singular value lower than `tol`.
        Default is 1e-12.
    singular_value_lower_bound : float, optional
        Lower bound of the largest singular value.
        If the largest singular value is lower than this
        value, it will send a warning to user.
        Default is 1e-6.

    Returns
    -------
    U : (m, r) double or complex ndarray
        Unitary matrix `U` in singular value decomposition, of shape (m, r).
    s : (r,) double or complex ndarray
        Singular value vector.
    Vh : (r, n) double or complex ndarray
        The transpose of unitary matrix `V` in singular value decomposition,
        of shape (r, n).
    """
    U, s, Vh = scipy.linalg.svd(A, full_matrices=False)
    first = s[0]
    if first < singular_value_lower_bound:
        warnings.warn("Too small singular values have been detected")
    # Returns the last index where
    cut = numpy.searchsorted(s / first < tol, True) - 1
    return U[:, :cut], s[:cut], Vh[:, :cut]
