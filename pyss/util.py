#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import scipy
import warnings
import functools
import itertools
import operator as op
from . import coefficient as coeff


def filter_k_function(method, gamma, rho, zeta, dzeta, n, k):
    """Summary

    Parameters
    ----------
    method : function
        Description
    gamma : double
        Description
    rho : double
        Description
    zeta : function
        Description
    dzeta : function
        Description
    n : int
        Description
    m : int
        Description

    Returns
    -------
    double -> double
        Filter function
    """
    zeta = np.vectorize(zeta)
    dzeta = np.vectorize(dzeta)

    radians = np.linspace(start=0, stop=np.pi*2, num=n, endpoint=True)
    ws      = method(n, 2*np.pi) * dzeta(radians) / (2 * np.pi * 1j)
    zetas   = zeta(radians)
    zs      = gamma + rho * zeta(radians)

    f_k = lambda eig: np.real(np.sum(rho * ws * (zetas ** k) / (zs - eig)))
    return np.vectorize(f_k)


def filter_function(method, gamma, rho, zeta, dzeta, n, m):
    f_k = lambda k: filter_k_function(method, gamma, rho, zeta, dzeta, n, k)

    def f(eig):
        acc = 0.0
        return reduce(lambda acc, k: acc + (f_k(k))(eig), range(0, m), 0)
    return f

def count_iterable(it):
    return sum(1 for i in it)
def integral(ys, h, samplep=2, contour=False):
    """
    Compute a definite integral.

    Parameters
    ----------
    ys : Num
    h : Float
    contour: Boolean

    Returns
    -------
    value : Float
    Value of integral.
    """
    n = len(ys)
    # numerical integral implemented by using trapezoid rule
    # int_a^b f(x) dx = h / 2 * sum_{k=1}^N ( f(x_{k+1}) + f(x_k) )
    # with the error term - (b-a)^3/(12N^2) * f''(c) where a < c < b
    ws = coeff.composite_newton_cotes_coeff(h, samplep, n, tr=None, contour=False)
    # TODO: Rewrite reduce with numpy
    return functools.reduce(op.add, map(op.mul, ws, ys))


def contour_integral(ys, h, samplep=2):
    """
    Compute a definite integral.

    Parameters
    ----------
    ys : Num
    h : Float
    contour: Boolean

    Returns
    -------
    value : Float
    Value of integral.
    """
    n = len(ys)
    # numerical integral implemented by using trapezoid rule
    # int_a^b f(x) dx = h / 2 * sum_{k=1}^N ( f(x_{k+1}) + f(x_k) )
    # with the error term - (b-a)^3/(12N^2) * f''(c) where a < c < b
    ws = coeff.composite_newton_cotes_coeff(h, samplep, n, tr=None, contour=True)
    # TODO: Rewrite reduce with numpy
    return functools.reduce(op.add, map(op.mul, ws, ys))


def trimmed_svd(A, tol=1e-12, singular_value_lower_bound=1e-6):
    """
    Singular value decomposition, with trimming too small singular values.

    Parameters
    ----------
    A : numpy.matrix
        A real or complex matrix of shape (m, n).
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
    U : (m, r) array
        Unitary matrix `U` in singular value decomposition, of shape (m, r).
    s : (r) array
        Singular value vector, of shape (r, r).
    Vh : (r, n) array
        The transpose of unitary matrix `V` in singular value decomposition,
        of shape (r, n).
    """
    U, s, Vh = numpy.linalg.svd(A, full_matrices=False)
    first = s[0]
    if first < singular_value_lower_bound:
        warnings.warn("Too small singular values have been detected")
    cut = count_iterable(itertools.takewhile(lambda x: x > tol, s / first))
    return U[:, :cut], s[:cut], Vh[:, :cut]


def rayleigh_ritz(A, B, U):
    """
    Rayleigh-Ritz procedure.
    Extended for general eigen value problems e.g. Ax = λBx.

    Parameters
    ----------
    A : (n, n) numpy.matrix
        A real or complex matrix whose eigenvalues and eigenvectors will be
        computed.
    B : (n, n) numpy.matrix
        Right-hand side
    U : numpy.matrix
        Unitary matrix `U` of `R = U*AU` of Rayleigh-Ritz procedure.

    Returns
    -------
    ws : (n,) double or complex ndarray
         Eigenvalues
    Vr : (n, n) double or complex ndarray
         The normalized right eigenvector corresponding to the eigenvalue
    """
    # 1. Compute an orthogonal basis U (n*m) approaximating the eigen space
    #    corresponding to m eigenvectors
    # 2. Compute R = U*AU
    # 3. Compute the eigenpairs (x, u) of R
    # 4. Return Eigenpairs (x, v) of Av = xBv where v = Uu
    UH = U.conj().T
    A = UH.dot(A.dot(U))
    B = UH.dot(B.dot(U))
    # Solve Av = lambda B v
    ws, V = scipy.linalg.eig(A, B)
    return ws, U.dot(V)


def shifted_rayleigh_ritz(A, B, U, shift):
    """
    Rayleigh-Ritz procedure.
    Extended for general eigen value problems with shifting.

    Parameters
    ----------
    A     : (n, n) numpy.matrix
            A real or complex matrix whose eigenvalues and eigenvectorswill be
            computed.
    B     : (n, n) numpy.matrix
            Right-hand side
    U     : (n, n) numpy.matrix
            Unitary matrix `U` of `R = U*AU` of Rayleigh-Ritz procedure.
    shift : double or complex number
            Shift of matrix A.

    Returns
    -------
    ws : (n,) double or complex ndarray
         Eigenvalues
    Vr : (n, n) double or complex ndarray
         The normalized right eigenvector corresponding to the eigenvalue
    """
    ws, V = rayleigh_ritz(A - shift*B, B, U)
    return ws + shift, V
