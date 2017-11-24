#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy


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
