#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from mpi4py import MPI


class TestMPI(unittest.TestCase):
    def test_size(self):
        world_comm = MPI.COMM_WORLD
        size = world_comm.Get_size()
        self.assertTrue(size > 0)
