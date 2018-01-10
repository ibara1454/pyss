# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import unittest
# import numpy as np
# import numpy.testing as npt
# from mpi4py import MPI
# from pyss.mpi import build_source, build_partial_moment


# def factorints(n):
#     """
#     Return all factors of integer `n`.
#     """
#     for i in (x + 1 for x in range(n)):
#         if n % i == 0:
#             yield i


# class TestPyss(npt.TestCase):
#     def helper_assert_same(self, x, comm):
#         """
#         Assert the value `x` in all nodes are the same.
#         """
#         size = comm.Get_size()
#         y = x / size
#         y = comm.reduce(y)
#         self.assert_almost_equal(x, y)

#     def test_build_source(self):
#         shape = (10, 5)
#         world = MPI.COMM_WORLD
#         size = world.Get_size()
#         for factor in factorints(size):
#             with self.subTest(size_n=factor):
#                 row_comm = world.Split(factor % i)
#                 col_comm = world.Split(factor // i)
#                 v = build_source(shape, col_comm)
#                 self.helper_assert_same(v, row_comm)
