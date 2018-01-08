#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
from pyss.util.contour import Circle


A = scipy.io.mmread("matrix/mhd4800a.mtx")
B = scipy.io.mmread("matrix/mhd4800b.mtx")
contour = Circle(center=-100, radius=5)
option = {
    'l': 50,
    'm': 5,
    'n': 12,
    'refinement': {
        'max_it': 1,
        'tol': 1e-6
    }
}
