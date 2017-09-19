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
from pyss.mpi.helper.option import replace_source, replace_solver
from pyss.algorithm import (
    trimmed_svd, shifted_rayleigh_ritz
)
from mpi4py import MPI

default_opt = {
    'l': 16,
    'm': 8,
    'n': 24,
    'refinement': {
        'max_it': 5,
        'tol': 1e-6
    },
    'quadrature': 1,
    'symmetric': False,
    'solve': 'linsolve',
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

    # TODO: Rename API replace_** names
    options.source = replace_source(options.source)
    options.solve = replace_solver(options.solve)

    return pyss_impl_rr(A, B, contour, options, comm)


def pyss_impl_rr(A, B, ctr, opt, comm):
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
    # Generate source matrix V
    V = build_source(A.shape[0], opt.l, opt, comm)
    S = build_moment(A, B, V, ctr, opt, comm)
    w, vr, res = rr_restrict_eig(A, B, S, ctr, comm)
    return w, vr, res


def build_source(n, m, opt, comm):
    """
    Generate source matrix on the root of given communicator.

    Parameters
    ----------
    n : int
        Size of row of source matrix.
    m : int
        Size of column of source matrix.
    opt : ()
        Pyss options.
    comm : MPI_Comm
        MPI communicator.

    Returns
    -------
    V : (n, m) numpy dense array
        Source matrix.
    """
    rank = comm.Get_rank()
    if rank == 0:
        V = opt.source(n, m)
    else:
        V = None
    return V


def rr_restrict_eig(A, B, S, ctr, comm):
    rank = comm.Get_rank()
    if rank == 0:
        U, _, _ = trimmed_svd(S)
        # Solve the reduced eigenvalue problem with Rayleigh-Ritz Procedure
        eigval, eigvec = shifted_rayleigh_ritz(A, B, U, shift=ctr.center)
        # Filtering eigen pairs whether which located in the contour
        eigval, eigvec = eig_pair_filter(eigval, eigvec, ctr.is_inside)
        # Calculate relative 2-norm for each eigen pairs, and find the maximum
        res = np.amax(eig_residul(A, B, eigval, eigvec))
    else:
        eigval, eigvec, res = [None, None, None]
    res = comm.bcast(res)
    return eigval, eigvec, res


def build_moment(A, B, V, ctr, opt, comm):
    rank = comm.Get_rank()
    index = int(rank / opt.n)
    # Create a communicator for first n (process 0 ~ n-1) to share V
    head_comm = comm.Split(0 if index == 0 else MPI.UNDEFINED)
    # Share source matrix V to each communicator which solves equations
    V = bcast_if_comm_is_not_null(V, head_comm)
    #
    Y = solve_at_quadrature_point(A, B, V, ctr, opt, comm)
    # Reduce matrix sum of each process
    S = reduce_integration(Y, ctr, opt, head_comm)
    return S


def bcast_if_comm_is_not_null(x, comm):
    """
    MPI broadcast. Broadcast object over given communicator only if the
    communicator is not null.

    Parameters
    ----------
    x : any object
        The object would shared over given communicator.
    comm : MPI_Comm
        MPI communicator. Could be a valid communicator or null communicator.
        The object `x` will be broadcast only if `comm` is not null.

    Returns
    -------
    x : any object | None
        Return the object `x` if `comm` is not MPI.COMM_NULL, or return None if
        `comm` is MPI.COMM_NULL.
    """
    if comm != MPI.COMM_NULL:
        x = comm.bcast(x)
    else:
        x = None
    return x


# TODO: Better rename API name
def solve_parallel(A, B, V, ctr, opt, comm):
    rank = comm.Get_rank()
    index = rank % opt.n
    sub_comm = comm.Split(index)
    z = ctr.func(ctr.domain_length * index / opt.n)
    # Sepatate source matrix V on the communicator which solves
    # linear equation
    V = opt.separater(V, sub_comm)
    # Solve linear equation paralleled
    # and gather the result to the process of rank = 0
    #
    # These matrices A and B could be pre-separated on each node
    # The statements z * B - A and B @ V would be calculated parallel
    # TODO: Better separation
    Y = opt.solve(z * B - A, B @ V, sub_comm)
    return Y


def reduce_integration(A, B, V, ctr, opt, comm):
    if comm != MPI.COMM_NULL:
        # Or do the multiplication before solving linear equation
        # by multiply the invert of w on the left hand side of equation
        # It could be done paralleled
        # w = w_i(quadrature, n, index)
        rank = comm.Get_rank()
        w = generate_weights_of_quadrature_points(ctr.df, opt.quadrature, opt.n)[rank]
        Y *= w
        S = np.hstack(Y * z ** k for z in range(opt.m))
        S = comm.reduce(S, op=MPI.SUM)
    else:
        S = None
    return S
