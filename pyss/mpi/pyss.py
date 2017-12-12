#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from attrdict import AttrDict
from pyss.util.generator import (
    generate_points_on_curve, generate_weights_of_quadrature_points
)
from pyss.util.analysis import eig_residul
from pyss.util.filter import eig_pair_filter
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
    # TODO: Throw exception while the size of `comm` does not match `n` *
    #       `opt.solver_comm_size`
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
    rank = comm.Get_rank()
    solve_comm = comm.Split(rank / opt.n)
    # Generate source matrix V
    V = build_source(A.shape[0], opt.l, solve_comm)
    S = build_moment(A, B, V, ctr, opt, comm)
    w, vr, res = rr_restrict_eig(A, B, S, ctr, comm)
    return w, vr, res


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
    solve_comm = comm.Split(rank / opt.n)

    z = ctr.func(ctr.domain_length * index / opt.n)
    # w = generate_weights_of_quadrature_points(ctr.df, opt.quadrature, opt.n)[index]
    w = (2 * np.pi / opt.n) * 2
    dz = ctr.df(ctr.domain_length * index / opt.n)
    Y = w * dz * opt.solve(z * B - A, B @ V, sub_comm)
    # Y = transform(Y)
    # Reduce matrix sum of each process
    S = reduce_integration(Y, ctr, opt)
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


def cal_value_on_quadrature_point(A, B, V, index, ctr, opt, comm):
    w = generate_weights_of_quadrature_points(ctr.df, opt.quadrature, opt.n)[index]
    z = ctr.func(ctr.domain_length * index / opt.n)
    dz = ctr.df(ctr.domain_length * index / opt.n)
    # Since the solution after calling the linear solver, only exist in the
    # node of rank 0 of communicator. Multiplying the weights to solution
    # on single node might be a large cost.
    # To minimize the cost, we multiply the weights to the right-hand side
    # of equation before calling the solving function
    return opt.solve(z * B - A, B @ V * (w * z * dz), comm)


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
        S = np.hstack(Y * z ** k for z in range(opt.m))
        S = comm.reduce(S, op=MPI.SUM)
    else:
        S = None
    return S
