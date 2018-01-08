#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from functools import partial
from attrdict import AttrDict
from pyss.util.generator import (
    generate_points_on_curve, generate_weights_of_quadrature_points
)
from pyss.util.analysis import eig_residul
from pyss.util.filter import eig_pair_filter
from pyss.standalone.helper.option import replace_source, replace_solver
from pyss.standalone.algorithm import (
    trimmed_svd, shifted_rayleigh_ritz
)


default_opt = {
    'l': 16,
    'm': 8,
    'n': 32,
    'refinement': {
        'max_it': 5,
        'tol': 1e-6
    },
    'quadrature': 1,
    'symmetric': False,
    'solver': 'linsolve',
    'source': 'random'
}


def solve(A, B, contour, l, m, n, executor):
    """
    Sakurai-Sugiura method, a paralleled generalized eigenpair solver.

    Sakurai-Sugiura method uses the contour integration and the complex
    moment to computes eigenvalues located inside a contour path on the
    complex plane.

    Parameters
    ----------
    A : (N, N) array_like
        The complex or real square matrix in the generalized eigenvalue
        problem.
    b : (N, N) array_like
        Right-hand side square matrix in the generalized eigenvalue problem.
    contour : pyss.util.contour.Curve
        The contour where the eigenvalues we desired located inside.
        `contour` is the instance of pyss.util.contour.Curve, which contains
        the imformation of contour path.
    executor : concurrent.futures.Executor
        Instance of executor. Used in calculating complex moment in ss-method.
    Returns
    -------
    w : (n,) array, where n <= M
        The eigenvalues.
    vr : (n, N) array, where n <= N
        The right eigenvectors. Each eigenvector vr[:,i] is corresponding to
        the eigenvalue w[i].
    """
    # Do not use class-base but function-base to implement this algorithm,
    # since functions have fewer side-effects, and are more familiar with MPI
    # parallelism

    source = replace_source('random')
    solver = replace_solver('linsolve')

    return pyss_impl_rr(A, B, contour, l, m, n, source, solver, executor)


def pyss_impl_rr(A, B, ctr, l, m, n, source, solver, executor):
    """
    The implementation part of Sakurai-Sugiura method using Rayleigh-Ritz
    procedure.

    In the main loop in this function, we use Rayleigh-Ritz procedure
    with refinement to implement the Sakurai-Sugiura method.
    After the complex moment S generated, we use S to filtering the original
    A and B, and evaluate the eigenvalues and eigenvectors with Rayleigh-Ritz
    procedure. If the residuals of eigenvalues and eigenvectors are not small
    enough, the complex moment S will be regenerate until the residual reaches
    tolerance.

    Parameters
    ----------
    A : (N, N) array_like
        The complex or real square matrix in the generalized eigenvalue
        problem.
    b : (N, N) array_like
        Right-hand side square matrix in the generalized eigenvalue problem.
    ctr : pyss.util.contour.Curve
        The contour where the eigenvalues we desired located inside.
        `contour` is the instance of pyss.util.contour.Curve, which contains
        the imformation of contour path.
    opt : attrdict.AttrDict
        The primary options of Pyss.
    source : function
        The generator of source matrix.
    solver : function
        The solver of linear equation of Ax = B.
    executor : concurrent.futures.Executor
        Instance of executor. Used in calculating complex moment in ss-method.

    Returns
    -------
    w : (n,) array, where n <= M
        The eigenvalues.
    vr : (n, N) array, where n <= N
        The right eigenvectors. Each eigenvector vr[:,i] is corresponding to
        the eigenvalue w[i].
    """
    # Initializing
    build_moment_with = moment_builder(A, B, ctr, m, n, solver, executor)
    # Build the first source matrix from given function
    V = source(A.shape[0], l)
    count = 0
    # TODO: rewrite while loop without side-effects
    # Do refinement until the residual is small enough
    # while count < opt.refinement.max_it and res > opt.refinement.tol:
    S = build_moment_with(V)
    U, _, _ = trimmed_svd(S)
    # Solve the reduced eigenvalue problem with Rayleigh-Ritz Procedure
    w, vr = shifted_rayleigh_ritz(A, B, U, shift=ctr.center)
    # Filtering eigen pairs whether which located in the contour
    w, vr = eig_pair_filter(w, vr, ctr.is_inside)
    # Calculate relative 2-norm for each eigen pairs, and find the maximum
    res = np.amax(eig_residul(A, B, w, vr))
    count += 1
    # Let S_0 be the source in next iteration
    # V = S[:, :l]
    return w, vr, res


def processing_with_solver(solver, args):
    """
    The procedure using by Executor in building complex moment.

    Parameters
    ----------
    solver : callable
        The linear solver of equation Ax = B.
        `solver` recieves two arguments of array-like objects, and returns
        a matrix.
    args : tuple of (complex, complex, (n, n) array-like, (n, n) array-like,
        (n, m) array-like)
        `args` the tuple of (w, z, A, B, V), the the arguments recieved from
        executor.map.
    """
    # Since the method which passed to executor.map must be Pickable,
    # processing_with_solver is defined in the top level in this module
    w, z, A, B, V = args
    return w * solver(z * B - A, B @ V)


def moment_builder(A, B, ctr, m, n, solver, executor):
    """
    The builder of complex moment. Calculus the complex moment by solving
    linear equations on each quadrature points on contour.

    Parameters
    ----------
    A : (N, N) array_like
        The complex or real square matrix in the generalized eigenvalue
        problem.
    b : (N, N) array_like
        Right-hand side square matrix in the generalized eigenvalue problem.
    ctr : pyss.util.contour.Curve
        The contour where the eigenvalues we desired located inside.
        `contour` is the instance of pyss.util.contour.Curve, which contains
        the imformation of contour path.
    opt : attrdict.AttrDict
        The primary options of Pyss.

    Returns
    -------
    S : (N, L*M) array
        Complex moment.
    """
    degree = 1
    ws = generate_weights_of_quadrature_points(ctr.df, degree, n)
    zs = generate_points_on_curve(ctr.func, n)
    proc = partial(processing_with_solver, solver)

    def build_moment_with(V):
        # Create processes for solving linear equations
        Ys = executor.map(proc, ((w, z, A, B, V) for w, z in zip(ws, zs)))
        # Since we will use Ys twice, create Ys explicit
        Ys = list(Ys)
        moment_k = lambda k: sum(z ** k * Y for (z, Y) in zip(zs, Ys))
        return np.hstack(moment_k(k) for k in range(m))
    return build_moment_with
