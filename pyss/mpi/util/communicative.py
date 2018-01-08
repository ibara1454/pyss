#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mpi4py import MPI


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
