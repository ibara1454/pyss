from mpi4py import MPI
import pyss.mpi
import pyss
import numpy
import scipy.io
from pyss.util.contour import Circle
import cProfile
import pstats
from profiler.profiler_mhd4800 import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
profiler = cProfile.Profile()
profiler.enable()
ws, vs, info = pyss.mpi.solve(A, B, contour, option, comm)
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('tottime').print_stats(10)
