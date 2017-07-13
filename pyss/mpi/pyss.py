#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from functools import partial
from attrdict import AttrDict
from pyss.helper.generator import (
    generate_points_on_curve, generate_weights_of_quadrature_points
)
from pyss.analysis import eig_residul
from pyss.helper.filter import eig_pair_filter
from pyss.helper.option import replace_source, replace_solver
from pyss.algorithm import (
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


def solve(A, B, contour, options, comm):
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
    options : dict
        The primary options of Pyss.
    comm : MPI_Comm
        MPI communicator.
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
    options = AttrDict({**default_opt, **options})

    source = replace_source(options.source)
    solver = replace_solver(options.solver)

    return pyss_impl_rr(A, B, contour, options, source, solver, executor)


def pyss_impl_rr(A, B, ctr, opt, source, solver, comm):
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
    comm : MPI_Comm
        MPI communicator.

    Returns
    -------
    w : (n,) array, where n <= M
        The eigenvalues.
    vr : (n, N) array, where n <= N
        The right eigenvectors. Each eigenvector vr[:,i] is corresponding to
        the eigenvalue w[i].
    """
    # Initializing
    build_moment_with = moment_builder(A, B, ctr, opt, solver, comm)
    # Build the first source matrix from given function
    V = source(A.shape[0], opt.l)
    res_iter = []
    res = float('inf')
    count = 0
    # TODO: rewrite while loop without side-effects
    # Do refinement until the residual is small enough
    while count < opt.refinement.max_it and res > opt.refinement.tol:
        S = build_moment_with(V)
        U, _, _ = trimmed_svd(S)
        # Solve the reduced eigenvalue problem with Rayleigh-Ritz Procedure
        w, vr = shifted_rayleigh_ritz(A, B, U, shift=ctr.center)
        # Filtering eigen pairs whether which located in the contour
        w, vr = eig_pair_filter(w, vr, ctr.is_inside)
        # Calculate relative 2-norm for each eigen pairs, and find the maximum
        res = np.amax(eig_residul(A, B, w, vr))
        res_iter = res_iter + [res]
        count += 1
        # Let S_0 be the source in next iteration
        V = S[:, :opt.l]
    # Make up all imformations into info
    info = {'residual': res_iter, 'iter_count': count}
    return w, vr, info


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


def moment_builder(A, B, ctr, opt, solver, executor):
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
    ws = generate_weights_of_quadrature_points(ctr.df, opt.quadrature, opt.n)
    zs = generate_points_on_curve(ctr.func, opt.n)
    proc = partial(processing_with_solver, solver)

    def build_moment_with(V):
        # Create processes for solving linear equations
        Ys = executor.map(proc, ((w, z, A, B, V) for w, z in zip(ws, zs)))
        # Since we will use Ys twice, create Ys explicit
        Ys = list(Ys)
        moment_k = lambda k: sum(z ** k * Y for (z, Y) in zip(zs, Ys))
        return np.hstack(moment_k(k) for k in range(opt.m))
    return build_moment_with
