#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.testing as npt
import pyss.coefficient as coeff

h = 0.1

class TestSSIntegral(npt.TestCase):
    def test_newton_cotes_coeff_points_2(self):
        # Sampling points = 2
        x = coeff.newton_cotes_coeff(h, 2)
        y = np.array([1, 1]) * 1/2 * h
        npt.assert_array_almost_equal(x, y)
    def test_newton_cotes_coeff_points_3(self):
        # Sampling points = 3
        x = coeff.newton_cotes_coeff(h, 3)
        y = np.array([1, 4, 1]) * 1/3 * h
        npt.assert_array_almost_equal(x, y)
    def test_newton_cotes_coeff_points_4(self):
        # Sampling points = 4
        x = coeff.newton_cotes_coeff(h, 4)
        y = np.array([1, 3, 3, 1]) * 3/8 * h
        npt.assert_array_almost_equal(x, y)
    def test_newton_cotes_coeff_points_7(self):
        # Sampling points = 7
        x = coeff.newton_cotes_coeff(h, 7)
        y = np.array([41, 216, 27, 272, 27, 216, 41]) * 1/140 * h
        npt.assert_array_almost_equal(x, y)
    def test_composite_newton_cotes_coeff_points_2(self):
        # line integral
        n = 13
        x = coeff.composite_newton_cotes_coeff(h, 2, n, tr=None, contour=False)
        y = p2_coeff(h, n)
        npt.assert_array_almost_equal(x, y)
        # contour integral
        n = 12
        x = coeff.composite_newton_cotes_coeff(h, 2, n, tr=None, contour=True)
        y = p2_coeff(h, n+1)
        y[0] = y[0] + y[n]
        y = y[0:n]
        npt.assert_array_almost_equal(x, y)
    def test_composite_newton_cotes_coeff_points_3(self):
        # line integral
        n = 13
        x = coeff.composite_newton_cotes_coeff(h, 3, n, tr=None, contour=False)
        y = p3_coeff(h, n)
        npt.assert_array_almost_equal(x, y)
        # contour integral
        n = 12
        x = coeff.composite_newton_cotes_coeff(h, 3, n, tr=None, contour=True)
        y = p3_coeff(h, n+1)
        y[0] = y[0] + y[n]
        y = y[0:n]
        npt.assert_array_almost_equal(x, y)
    def test_composite_newton_cotes_coeff_points_4(self):
        # line integral
        n = 13
        x = coeff.composite_newton_cotes_coeff(h, 4, n, tr=None, contour=False)
        y = p4_coeff(h, n)
        npt.assert_array_almost_equal(x, y)
        # contour integral
        n = 12
        x = coeff.composite_newton_cotes_coeff(h, 4, n, tr=None, contour=True)
        y = p4_coeff(h, n+1)
        y[0] = y[0] + y[n]
        y = y[0:n]
        npt.assert_array_almost_equal(x, y)
    def test_composite_newton_cotes_coeff_points_7(self):
        n = 13
        x = coeff.composite_newton_cotes_coeff(h, 7, n, tr=None, contour=False)
        y = p7_coeff(h, n)
        npt.assert_array_almost_equal(x, y)
        # contour integral
        n = 12
        x = coeff.composite_newton_cotes_coeff(h, 7, n, tr=None, contour=True)
        y = p7_coeff(h, n+1)
        y[0] = y[0] + y[n]
        y = y[0:n]
        npt.assert_array_almost_equal(x, y)

def p2_coeff(h, n):
    ws = np.empty(n)
    for i in range(n):
        if i == 0 or i == (n-1):
            ws[i] = 1.0
        else:
            ws[i] = 2.0
    return 1/2 * h * ws

def p3_coeff(h, n): 
    ws = np.empty(n)
    for i in range(n):
        if i == 0 or i == (n-1):
            ws[i] = 1.0
        elif i % 2 == 0:
            ws[i] = 2.0
        else:
            ws[i] = 4.0
    return 1/3 * h * ws

def p4_coeff(h, n):
    ws = np.empty(n)
    for i in range(n):
        if i == 0 or i == (n-1):
            ws[i] = 1.0
        elif i % 3 == 0:
            ws[i] = 2.0
        else:
            ws[i] = 3.0
    return 3/8 * h * ws

def p5_coeff(h, n):
    ws = np.empty(n)
    for i in range(n):
        if i == 0 or i == (n-1):
            ws[i] = 7.0
        elif i % 4 == 0:
            ws[i] = 14.0
        elif i % 4 == 2:
            ws[i] = 12.0
        else:
            ws[i] = 32.0
    return 2/45 * h * ws

def p7_coeff(h, n):
    ws = np.empty(n)
    for i in range(n):
        if i == 0 or i == (n-1):
            ws[i] = 41.0
        elif i % 6 == 0:
            ws[i] = 82.0
        elif i % 6 == 1 or i % 6 == 5:
            ws[i] = 216.0
        elif i % 6 == 2 or i % 6 == 4:
            ws[i] = 27.0
        else:
            ws[i] = 272.0
    return 1/140 * h * ws
