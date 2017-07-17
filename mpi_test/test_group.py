#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mpi4py import MPI

world_comm = MPI.COMM_WORLD
world_rank = world_comm.Get_rank()
world_size = world_comm.Get_size()

world_group = world_comm.Get_group()

n = 7
ranks = [1, 2, 3, 5, 7, 11, 13]

prime_group = world_group.Incl(ranks)
prime_comm = MPI.Comm.Create(world_comm, prime_group)

if MPI.COMM_NULL != prime_comm:
    prime_rank = prime_comm.Get_rank()
    prime_size = prime_comm.Get_size()
else:
    prime_rank = -1
    prime_size = -1

print("WORLD RANK/SIZE: {}/{}\tPRIME RANK/SIZE: {}/{}, ROOT={}\n".format(
    world_rank, world_size, prime_rank, prime_size, MPI.ROOT
))

if MPI.COMM_NULL != prime_comm:
    prime_comm.Free()
prime_group.Free()
world_group.Free()
