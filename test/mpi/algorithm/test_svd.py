#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import numpy.random
import numpy.testing as npt
import scipy
import scipy.linalg
import pyss.mpi.algorithm
import itertools
from mpi4py import MPI

decimal_low = 2
decimal_high = 4


def generate_real_common_test_matrix(m, n):
    return numpy.random.rand(m, n)


def generate_complex_common_test_matrix(m, n):
    return generate_real_common_test_matrix(m, n) + \
        1j * generate_real_common_test_matrix(m, n)


def generate_complex_hard_test_matrix(m, n):
    a = generate_complex_common_test_matrix(m, n)
    s = a.T @ a
    return a @ s


class TestSvd(npt.TestCase):
    def setUp(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # Set random seet by rank of communicator
        numpy.random.seed(rank)
        self.comm = comm

    def test_real_common(self):
        ms = [500, 1000, 2000]
        ns = [200]
        for (m, n) in itertools.product(ms, ns):
            with self.subTest(m=m, n=n):
                a = generate_real_common_test_matrix(m, n)
                U, s, Vh = pyss.mpi.algorithm.svd(a, self.comm)
                S = numpy.diag(s)
                approx = U @ S @ Vh
                npt.assert_array_almost_equal(approx, a, decimal=decimal_high)

    def test_complex_common(self):
        ms = [500, 1000, 2000]
        ns = [200]
        for (m, n) in itertools.product(ms, ns):
            with self.subTest(m=m, n=n):
                a = generate_complex_common_test_matrix(m, n)
                U, s, Vh = pyss.mpi.algorithm.svd(a, self.comm)
                S = numpy.diag(s)
                approx = U @ S @ Vh
                npt.assert_array_almost_equal(approx, a, decimal=decimal_high)

    def test_complex_hard(self):
        ms = [500, 2000]
        ns = [200]
        for (m, n) in itertools.product(ms, ns):
            with self.subTest(m=m, n=n):
                a = generate_complex_hard_test_matrix(m, n)
                U, s, Vh = pyss.mpi.algorithm.svd(a, self.comm)
                S = numpy.diag(s)
                approx = U @ S @ Vh
                npt.assert_array_almost_equal(approx, a, decimal=decimal_low)
