#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy
import scipy.io
import pyss
import cProfile
import pstats

A = scipy.io.mmread("matrix/bcsstk11.mtx")
B = scipy.io.mmread("matrix/bcsstm11.mtx")

L = 16
M = 8
N = 16

center = 1000
radius = 500
profiler = cProfile.Profile()
profiler.enable()
eigvals, eigvecs = pyss.solve(A, B, L, M, N, center, radius)
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('tottime').print_stats()