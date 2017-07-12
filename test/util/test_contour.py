#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from numpy import sin, cos, pi, exp
import numpy.testing as npt
from pyss.util.contour import Curve, Ellipse, Circle

test_value = [2 * x / 100 * pi for x in range(0, 100)]


class TestContour(npt.TestCase):
    cases = [
        {
            'curve': lambda x: 3 + 4 * exp(x * 1j),
            'df': lambda x: 1j * 4 * exp(x * 1j)
        }
    ]

    def test_df(self):
        for case in self.cases:
            curve = Curve(func=case['curve'])
            for v in test_value:
                npt.assert_almost_equal(case['df'](v), curve.df(v))


class TestEllipse(npt.TestCase):
    cases = [
        {
            'real': 2.0,
            'imag': 7.0,
            'shift': 5.0,
            'rotate': 0
        },
        {
            'real': 7.0,
            'imag': 2.0,
            'shift': 5.0 + 10j,
            'rotate': 0
        },
        {
            'real': 2.0,
            'imag': 7.0,
            'shift': 10.0 + 2.3j,
            'rotate': pi / 2
        },
        {
            'real': 7.0,
            'imag': 2.0,
            'shift': 5.0 + 1.0j,
            'rotate': 4 * pi / 3
        },
    ]

    def test_df(self):
        for case in self.cases:
            re = case['real']
            im = case['imag']
            shift = case['shift']
            rot = case['rotate']
            df = lambda x: (-re * sin(x) + im * 1j * cos(x)) * exp(1j * rot)
            ellipse = Ellipse(re, im, shift, rot)
            for v in test_value:
                npt.assert_almost_equal(df(v), ellipse.df(v))


class TestCircle(npt.TestCase):
    cases = [
        {
            'center': 1,
            'radius': 1
        },
        {
            'center': 10 * 5j,
            'radius': 2.5
        },
        {
            'center': 8j,
            'radius': 1000
        }
    ]

    def test_df(self):
        for case in self.cases:
            center = case['center']
            radius = case['radius']
            df = lambda x: radius * 1j * exp(1j * x)
            circle = Circle(center, radius)
            for v in test_value:
                npt.assert_almost_equal(df(v), circle.df(v))
