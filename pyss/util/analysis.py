#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg


def norm_along_column(a, ord=2):
    norm = lambda x: np.linalg.norm(x, ord=ord)
    return np.apply_along_axis(norm, 0, a)


def eig_residul(a, b, x, v, rel=True):
    av = a @ v
    bv = b @ v
    rs = norm_along_column(av - x * bv)
    if rel:
        return rs / (norm_along_column(av) + np.abs(x) * norm_along_column(bv))
    else:
        return rs
