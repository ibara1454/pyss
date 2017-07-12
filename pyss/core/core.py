#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import functools
import operator as op
from pyss.core import coefficient as coeff


def integral(ys, h, degree=1):
    """
    Compute a definite integration using closed composite Newton-Cotes
    quadrature. e.g. samplep = 2 for using the composite trapezoid rule,
    samplep = 3 for using composite simpson's rule.

    Parameters
    ----------
    ys : (n, ) array_like
        Each quadrature points on given path. The length of `ys` must be a
        multiple of (`samplep` - 1).
    h  : Float
        The interval of each quadrature points.
    degree : int
        Degree of interpolate polynomial which used in each newton-cotes
        quadrature rules.

    Returns
    -------
    value : Float
        Numerical integration.
    """
    if not hasattr(ys, '__len__'):
        ys = list(ys)
    n = len(ys)
    # int_on_contour f(x) dx = h / 2 * sum_{k=0}^{N-1} ( w_i * y_i )
    ws = coeff.composite_newton_cotes_coeff(h, degree, n, tr=None, contour=False)
    # TODO: Rewrite reduce with numpy
    return functools.reduce(op.add, map(op.mul, ws, ys))


def contour_integral(ys, h, degree=1):
    """
    Compute a definite contour integration using closed composite Newton-Cotes
    quadrature. e.g. samplep = 2 for using the composite trapezoid rule,
    samplep = 3 for using composite simpson's rule.

    Parameters
    ----------
    ys : (n, ) array_like
        Each quadrature points on contour. The length of `ys` must be a
        multiple of (`samplep` - 1).
    h  : Float
        The interval of each quadrature points.
    degree : int
        Degree of interpolate polynomial which used in each newton-cotes
        quadrature rules.

    Returns
    -------
    value : Float
        Numerical integration.
    """
    if not hasattr(ys, '__len__'):
        ys = list(ys)
    n = len(ys)
    # int_on_contour f(x) dx = h / 2 * sum_{k=0}^{N-1} ( w_i * y_i )
    ws = coeff.composite_newton_cotes_coeff(h, degree, n, tr=None, contour=True)
    # TODO: Rewrite reduce with numpy
    return functools.reduce(op.add, map(op.mul, ws, ys))
