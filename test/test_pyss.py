#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest
import numpy
import numpy.testing as npt
import scipy.io
import pyss
from pyss.util.contour import Circle, Ellipse
from concurrent.futures import ProcessPoolExecutor

decimal = 5


class TestPyss(npt.TestCase):
    @unittest.skipIf('--quick' in sys.argv, 'Large amount computations')
    def test_pyss_size_1473(self):
        # A, B are Real matries
        D = scipy.io.mmread("matrix/bcsst11eig.mtx")
        A = scipy.io.mmread("matrix/bcsstk11.mtx")
        B = scipy.io.mmread("matrix/bcsstm11.mtx")
        contour = Ellipse(real=200, imag=0.3, shift=900)
        D = D[contour.is_inside(D)]
        option = {'l': 8, 'm': 2, 'n': 12}
        with ProcessPoolExecutor() as executor:
            ws, vs, info = pyss.solve(A, B, contour, option, executor)
        ws = numpy.sort(ws)
        npt.assert_array_almost_equal(ws, D, decimal=decimal)

    @unittest.skipIf('--quick' in sys.argv, 'Large amount computations')
    def test_pyss_size_4800(self):
        # A, B are Real matries. 16 eigenvalues in contour
        D = scipy.io.mmread("matrix/mhd4800eig.mtx")
        A = scipy.io.mmread("matrix/mhd4800a.mtx")
        B = scipy.io.mmread("matrix/mhd4800b.mtx")
        contour = Circle(center=-100, radius=5)
        D = D[contour.is_inside(D)]
        option = {'l': 20, 'm': 5, 'n': 12}
        with ProcessPoolExecutor() as executor:
            ws, vs, info = pyss.solve(A, B, contour, option, executor)
        ws = numpy.sort(ws)
        npt.assert_array_almost_equal(ws, D, decimal=decimal)
