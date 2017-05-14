#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import numpy.testing as npt
import scipy
import scipy.io
import scipy.sparse
import scipy.sparse.linalg
import pyss

# A = scipy.io.mmread("bcsstk11.mtx")
# B = scipy.io.mmread("bcsstm11.mtx")

class TestPyss(npt.TestCase):
    def test_pyss_1(self):
        A = scipy.sparse.diags(numpy.arange(0.01, 10.01, 0.10))
        B = scipy.sparse.eye(100)
        L = 10
        M = 8
        N = 12
        center = 0
        radius = 1
        eigvals, eigvecs = pyss.solve(A, B, L, M, N, center, radius)
        print(numpy.sort(eigvals))
