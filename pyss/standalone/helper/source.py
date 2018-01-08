#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import numpy.random


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
    # TODO: Another way to make the source matrix fixed.
    # Make the source matrix fixed. To make sure unit test
    # returns the same result each times.
    numpy.random.rand(0)
    return numpy.random.rand(rows, cols)
