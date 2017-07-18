from mpi4py import MPI
import pyss.mpi
import pyss
import numpy
import scipy.io
from pyss.util.contour import Circle
import cProfile
import pstats

A = scipy.io.mmread("matrix/wathen100.mtx")
B = scipy.sparse.eye(30401)
option = {
    'l': 10,
    'm': 2,
    'n': 12,
    'refinement': {
        'max_it': 1,
        'tol': 1e-6
    }
}
contour = Circle(center=1, radius=0.1)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ws, vs, info = pyss.mpi.solve(A, B, contour, option, comm)
