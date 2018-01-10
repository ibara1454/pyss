#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

A = np.arange(rank * 4.0, rank * 4.0 + 4.0).reshape((2, 2))
print(A)
if rank == 0:
    recvbuf = np.empty((2, size * 2))
else:
    recvbuf = None

comm.Gather(A, recvbuf)

if rank == 0:
    print(recvbuf)
