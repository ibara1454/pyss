#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
from pyss.util.contour import Circle


A = scipy.io.mmread("matrix/wathen100.mtx")
B = scipy.sparse.eye(30401)
contour = Circle(center=1, radius=0.1)
option = {
    'l': 100,
    'm': 2,
    'n': 12,
    'refinement': {
        'max_it': 1,
        'tol': 1e-6
    }
}
