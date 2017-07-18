#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cProfile
import pstats
import numpy
import scipy
import scipy.io
from concurrent.futures import ProcessPoolExecutor
from pyss.util.contour import Circle
import pyss


def run():
    A = scipy.io.mmread("matrix/mhd4800a.mtx")
    B = scipy.io.mmread("matrix/mhd4800b.mtx")
    contour = Circle(center=-100, radius=5)
    option = {
        'l': 20,
        'm': 5,
        'n': 12,
        'refinement': {
            'max_it': 1,
            'tol': 1e-6
        }
    }
    for p in range(1, 1 + 8):
        print("==================Processes = {}==================".format(p))
        profiler = cProfile.Profile()
        profiler.enable()
        with ProcessPoolExecutor(max_workers=p) as executor:
            ws, vs, info = pyss.solve(A, B, contour, option, executor)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('tottime').print_stats(10)
