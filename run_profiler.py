#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concurrent.futures import ProcessPoolExecutor
import cProfile
import pstats

from profiler.profiler_wathen100 import *
import pyss

if __name__ == '__main__':
    for p in [24, 12, 8, 4, 2, 1]:
        print("==================Processes = {}==================".format(p))
        profiler = cProfile.Profile()
        profiler.enable()
        with ProcessPoolExecutor(max_workers=p) as executor:
            ws, vs, info = pyss.solve(A, B, contour, option, executor)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('tottime').print_stats(6)
