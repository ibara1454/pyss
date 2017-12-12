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

decimal = 4


def generate_real_test_matrix(m, n, k):
    return numpy.random.rand(m, n)


def generate_complex_test_matrix(m, n, k):
    return numpy.random.rand(m, n) + 1j * np.random.rand(m, n)


class TestSvd(npt.TestCase):
    def test_real_common(self):
        ms = [200, 400, 1000]
        ns = [200]
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        for (m, n) in itertools.product(ms, ns):
            with self.subTest(m=m, n=n):
                a = generate_real_test_matrix(m, n, rank)
                U, s, Vh = pyss.mpi.algorithm.svd(a, comm)
                S = numpy.diag(s)
                approx = U @ S @ Vh
                npt.assert_array_almost_equal(approx, a, decimal=decimal)

    def test_complex_common(self):
        ms = [200, 400, 1000]
        ns = [200]
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        for (m, n) in itertools.product(ms, ns):
            with self.subTest(m=m, n=n):
                a = generate_complex_test_matrix(m, n, rank)
                U, s, Vh = pyss.mpi.algorithm.svd(a, comm)
                S = numpy.diag(s)
                approx = U @ S @ Vh
                npt.assert_array_almost_equal(approx, a, decimal=decimal)
