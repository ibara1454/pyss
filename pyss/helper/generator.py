#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from pyss.core.coefficient import composite_newton_cotes_coeff


def generate_points_on_curve(f, n):
    xs = generate_points_on_curcumference(n)
    radian2cartesian = numpy.vectorize(f)
    return radian2cartesian(xs)


def generate_points_on_curcumference(n):
    return numpy.arange(n) * 2 * numpy.pi / n


def generate_weights_of_quadrature_points(df, degree, n):
    h = 2 * numpy.pi / n
    ws = composite_newton_cotes_coeff(h, degree, n, contour=True)
    dfs = generate_points_on_curve(df, n)
    return ws * dfs


def random_source_matrix(rows, cols):
    """
    Build a source matrix with size (rows, cols), by using random generator.

    Parameters
    ----------
    rows : int
        Size of rows.
    cols : int
        Size of columns.

    Returns
    -------
    V : (rows, cols) array
        Source matrix, of shape (rows, cols).
    """
    return numpy.random.rand(rows, cols)
