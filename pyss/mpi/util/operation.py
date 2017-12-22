#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Module of matrix and vector operations on MPI communicator.

The naming rule of functions follows.
h : horizontal distributed matrix or vecters.
v : vertical distributed matrix or vecters.
g : grid distributed matrix.
"""


def hv(h, v, comm):
    """
    Multiplication of
    `m = h v`
    where `h` is a horizontal distributed matrix (vector) on communicator,
    `v` is a vertical distributed matrix (vector) on communicator,
    and `m` is the result of multiplication

    Parameters
    ----------
    h : (M, K) array_like
        Distributed matrix (vector). The matrix (vector) is distributed on
        given communicator `comm` horizontally. In each node of `comm`, `h`
        is with the shape (M, K).
    v : (K, N) array_like
        Distributed matrix (vector). The matrix (vector) is distributed on
        given communicator `comm` vertically. In each node of `comm`, `v`
        is with the shape (K, N).
    comm : MPI_comm
        MPI communicator.

    Returns
    -------
    m : (M, N) array_like
        The result of multiplication. This matrix (vector) is distributed on
        given communicator `comm`. In each node of `comm`, `m` describe the
        same value.
    """
    return comm.allreduce(h @ v)
