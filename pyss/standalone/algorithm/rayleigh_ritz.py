#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.sparse.linalg


def rayleigh_ritz(A, B, U):
    """
    Rayleigh-Ritz procedure.
    Extended for general eigen value problems e.g. Ax = λBx.

    Parameters
    ----------
    A : (n, n) array_like
        A real or complex matrix of shape (n, n).
        Matrix A in eigen value problems Ax = λBx.
    B : (n, n) array_like
        A real or complex matrix of shape (n, n).
        Matrix B in eigen value problems Ax = λBx.
    U : (n, m) array_like
        Unitary matrix `U` of `R = U*AU` of Rayleigh-Ritz procedure.

    Returns
    -------
    ws : (m,) array
        Eigenvalues
    Vr : (n, m) array
        The normalized right eigenvector corresponding to the eigenvalue
    """
    # 1. Compute an orthogonal basis U (n*m) approaximating the eigen space
    #    corresponding to m eigenvectors
    # 2. Compute R = U*AU
    # 3. Compute the eigenpairs (x, u) of R
    # 4. Return Eigenpairs (x, v) of Av = xBv where v = Uu
    UH = U.conj().T
    A = UH @ (A @ U)
    B = UH @ (B @ U)
    # Solve Av = lambda B v
    ws, V = scipy.linalg.eig(A, B)
    return ws, U @ V


def shifted_rayleigh_ritz(A, B, U, shift):
    """
    Rayleigh-Ritz procedure.
    Extended for general eigen value problems with shifting.

    Parameters
    ----------
    A : (n, n) array_like
        A real or complex matrix of shape (n, n).
        Matrix A in eigen value problems Ax = λBx.
    B : (n, n) array_like
        A real or complex matrix of shape (n, n).
        Matrix B in eigen value problems Ax = λBx.
    U : (n, m) array_like
        Unitary matrix `U` of `R = U*AU` of Rayleigh-Ritz procedure.
    shift : float or complex
        Shift of matrix A.

    Returns
    -------
    ws : (m,) array
        Eigenvalues
    Vr : (n, m) array
        The normalized right eigenvector corresponding to the eigenvalue
    """
    ws, V = rayleigh_ritz(A - shift * B, B, U)
    return ws + shift, V
