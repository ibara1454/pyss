#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI

world = MPI.COMM_WORLD
world_size = world.Get_size()
world_rank = world.Get_rank()

if world_rank == 0:
    A = np.arange(9).reshape((3, 3))
else:
    A = np.empty((3, 3), dtype='i')

A = world.bcast(A)

print(A)
