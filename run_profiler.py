#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concurrent.futures import ProcessPoolExecutor
import cProfile
import pstats

from profiler.profiler_mhd4800 import *
import pyss

if __name__ == '__main__':
    for p in range(1, 1 + 8):
        print("==================Processes = {}==================".format(p))
        profiler = cProfile.Profile()
        profiler.enable()
        with ProcessPoolExecutor(max_workers=p) as executor:
            ws, vs, info = pyss.solve(A, B, contour, option, executor)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('tottime').print_stats(6)
