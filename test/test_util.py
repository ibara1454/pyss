#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import numpy.testing as npt
import pyss.util as util

a = 1
b = 2
h = 0.0001
prec = 3

class TestUtil(npt.TestCase):
    def test_integral(self): 
        f = numpy.vectorize(lambda x: x ** 2)
        xs = numpy.arange(a, b, h)
        ys = f(xs)
        ans = util.integral(ys, h)
        npt.assert_almost_equal(ans, 7/3, decimal=prec)