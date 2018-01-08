#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random
from pyss.mpi.algorithm import svd_low_rank, rayleigh_ritz
from pyss.util.contour import inside_filter

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


def solve(l_a, l_b, contour, l, m, n, solver, cv, comm):
    """
    Sakurai-Sugiura method, paralleled generalized eigenpair solver on
    MPI parallelism.

    Sakurai-Sugiura method uses contour integration and complex moment
    to computes eigenvalues located inside a contour path on the
    complex plane.

    ====================================================================

    In this implementation, it splits given communicator `comm` into `n`
    lower size sub-communicators `scomm`s (with size `ss = s / n` where
    `s` denotes the size of `comm`, hence `n` should be a factor of `s`)
    to compute each complex moment parallelly.

    The matrix `a` and `b` in general eigenvalue problem `ax = λbx`
    could be either numpy's ndarray or any matrix format distributed on
    `scomm`, but user should define matrix loaders `l_a` and `l_b` to
    load matrix after spliting `comm` into `scomm`, and matrix-matrix
    multiplication `cv` on `scomm`.

    User should also defined the solver `solver` of block linear equation
    `ax = v` on `scomm`, where the whole `v` on `scomm` is with shape
    (N, l), and is with shape (N / ss, l) on each node on `scomm`.

    Parameters
    ----------
    l_a : Callable object with type `l_a(scomm: MPI_Comm) -> (N, N) array_like`
        Matrix Loader of matrix `a` where `a` is a complex or real square
        matrix with shape (N, N) in the generalized eigenvalue problem.
    l_a : Callable object with type `l_a(scomm: MPI_Comm) -> (N, N) array_like`
        Matrix Loader of matrix `b` where `b` is right-hand side square matrix
        with shape (N, N) in the generalized eigenvalue problem.
    contour : pyss.util.contour.Curve
        The contour where the eigenvalues we desired located inside.
        `contour` is the instance of pyss.util.contour.Curve, which contains
        the imformation of contour path.
    l : Integer
        Size of source matrix. The whole source matrix is with shape (N, l)
        on `scomm`. And is with shape (N / ss, l) on each node of `scomm`.
    m : Integer
        Number of complex moment.
    n : Integer
        Number of quadrature points of numerical integrations. The
        approximation error of numerical integration is inverse proportion
        to `n`.
    solver : Callable object with type
        `solver(a: matrix, v: ndarray, scomm: MPI_Comm) -> ndarray`.
        The solver of linear equation `ax = v` in communicator `scomm`,
        where `a` is the matrix given by user, `v` is a ndarray with
        shape (N / ss, l), and `ss` is the size of `scomm`. Notice that
        solver should return the same data type and same shape to `v`.
    cv : Callable object with type
        `cv(c: (N, N) array_like, v: ndarray, scomm: MPI_Comm) -> ndarray`.
        The function of multiplying of matrix `c` and matrix `v` on `scomm`,
        where `c` is the matrix loaded by either `l_a` or `l_b`, and `v` is
        ndarray with shape (N / rank, l) on each node of `scomm`.
        The return value of `cv` should be a ndarray, and with the same shape
        of `v`.
    comm : MPI_Comm
        MPI communicator. The size of `comm` should be a factor of `n`.

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
    # options = AttrDict({**default_opt, **options})

    # TODO: Throw exception while the size of `comm` does not match `n` *
    #       `opt.solver_comm_size`
    return pyss_impl_rr(l_a, l_b, contour, l, m, n, solver, cv, comm)


def pyss_impl_rr(l_a, l_b, contour, l, m, n, solver, cv, comm):
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
    """
    # Initializations
    rank = comm.Get_rank()
    row_index = rank // opt.n
    col_index = rank % opt.n
    ccomm = comm.Split(row_index)
    rcomm = comm.Split(col_index)

    a = l_a(ccomm)
    b = l_b(ccomm)
    v = build_source((A.shape[0], l), ccomm)
    # Build the parts of moment on each `ccomm`
    s = build_partial_moment(a, b, v, ctr, col_index, m, solver, cv, ccomm)
    # Build the same moment on each `ccomm` (share data column-wisely)
    s = build_total_moment(s, rcomm)
    # Calculate eigenpairs on each `ccomm`
    w, vr, res = rr_restrict_eig(s, a, b, ctr, cv, ccomm)
    return w, vr, res


def build_source(shape, comm):
    size = comm.Get_size()
    rank = comm.Get_rank()
    # Fix the random seed to ensure the source matrix is unique on each node
    # of `comm`. But the same matrix on each node of the same rank (on other
    # communicators)
    # TODO: try use id(comm) * rank
    np.random.seed(rank)
    # Generate random matrix `v`, which is distributed on `comm` vertically
    v = np.random.rand(shape[0] / size, shape[1])
    return v


def build_partial_moment(a, b, v, ctr, index, m, solver, cv, comm):
    z = ctr.func(ctr.domain_length * index / opt.n)
    df = ctr.df(ctr.domain_length * index / opt.n)
    w = (2 * np.pi / opt.n) * 2
    x = (w * df) * solver(z * b - a, cv(b, v, comm), comm)
    # Build m (partial) moment and combine along row axis
    # Use generator instead of build matrix directly
    s = np.hstack((z ** k) * x for k in range(m))
    return s


def build_total_moment(s, comm):
    return comm.reduce(s)


def rr_restrict_eig(s, a, b, ctr, cv, comm):
    p, _, _ = svd_low_rank(s, 3, comm)
    # Solve the reduced eigenvalue problem with Rayleigh-Ritz Procedure
    # TODO: implements the shifted version of Rayleigh-Ritz
    eigval, eigvec = rayleigh_ritz(p, a, b, cv, comm)
    # Filtering eigen pairs whether which located in the contour
    eigval, eigvec = inside_filter(eigval, eigvec, ctr)
    # Calculate relative 2-norm for each eigen pairs, and find the maximum
    # res = np.amax(eig_residul(A, B, eigval, eigvec))
    return eigval, eigvec, None
