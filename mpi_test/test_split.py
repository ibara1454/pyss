#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from mpi4py import MPI

world = MPI.COMM_WORLD
world_size = world.Get_size()
world_rank = world.Get_rank()

if world_rank == 0:
    A = 100
else:
    A = None

color = world_rank % 2
row_comm = world.Split(color=color, key=world_rank)
row_rank = row_comm.Get_rank()
row_size = row_comm.Get_size()

color = math.floor(world_rank / 2)
col_comm = world.Split(color=color, key=world.rank)
col_rank = col_comm.Get_rank()
col_size = col_comm.Get_size()

if col_rank == 0:
    A = row_comm.bcast(A, root=0)

print("WORLD RANK/SIZE {}/{}\tROW RANK/SIZE: {}/{}\tCOL RANK/SIZE {}/{}\tA = {}".format(
    world_rank, world_size, row_rank, row_size, col_rank, col_size, A
))