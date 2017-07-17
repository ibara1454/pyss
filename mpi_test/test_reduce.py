#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

A = np.arange(rank * 4.0, rank * 4.0 + 4.0).reshape((2, 2))

B = comm.reduce(A)
print(B)
