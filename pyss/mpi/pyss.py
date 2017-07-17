#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import floor
from functools import partial
from attrdict import AttrDict
from pyss.helper.generator import (
    generate_points_on_curve, generate_weights_of_quadrature_points
)
from pyss.analysis import eig_residul
from pyss.helper.filter import eig_pair_filter
from pyss.mpi.helper.option import replace_source, replace_solver
from pyss.algorithm import (
    trimmed_svd, shifted_rayleigh_ritz
)
from mpi4py import MPI

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

    return pyss_impl_rr(A, B, contour, options, source, solver, comm)


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
    rank = comm.Get_rank()
    # Generate source matrix V
    V = source(A.shape[0], opt.l) if rank == 0 else None
    S = build_moment(A, B, V, ctr, opt, solver, comm)
    w, vr, res = restrict_eig(A, B, S, ctr, comm)
    if rank == 0:
        print(res)
        print(w)
    return w, vr, res


def restrict_eig(A, B, S, ctr, comm):
    if comm.Get_rank() == 0:
        U, _, _ = trimmed_svd(S)
        # Solve the reduced eigenvalue problem with Rayleigh-Ritz Procedure
        eigval, eigvec = shifted_rayleigh_ritz(A, B, U, shift=ctr.center)
        # Filtering eigen pairs whether which located in the contour
        eigval, eigvec = eig_pair_filter(eigval, eigvec, ctr.is_inside)
        # Calculate relative 2-norm for each eigen pairs, and find the maximum
        res = np.amax(eig_residul(A, B, eigval, eigvec))
    else:
        eigval = None
        eigvec = None
        res = None
    res = comm.bcast(res)
    return eigval, eigvec, res


def build_moment(A, B, V, ctr, opt, solver, comm):
    rank = comm.Get_rank()
    # Create a communicator (process 0 ~ n-1) to share V
    per_n = floor(rank / opt.n)
    y_comm = comm.Split(0 if per_n == 0 else MPI.UNDEFINED)
    # Share source matrix V over y_comm
    if y_comm != MPI.COMM_NULL:
        V = y_comm.bcast(V)
    index = rank % opt.n
    w = generate_weights_of_quadrature_points(ctr.df, opt.quadrature, opt.n)[index]
    z = generate_points_on_curve(ctr.func, opt.n)[index]
    solve_comm = comm.Split(rank % opt.n)
    # TODO: better parallelization for linear solver
    V = solve_comm.bcast(V)
    Y = solver(w * (z * B - A), B @ V, solve_comm)
    # Reduce matrix sum of each process
    S = y_comm.reduce(Y) if y_comm != MPI.COMM_NULL else None
    Y = z * Y if y_comm != MPI.COMM_NULL else None
    S2 = y_comm.reduce(Y) if y_comm != MPI.COMM_NULL else None
    print(S)

    return np.hstack([S, S2])
