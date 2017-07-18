import numpy
import scipy.io
import pyss.mpi
from pyss.util.contour import Ellipse
from mpi4py import MPI

A = scipy.io.mmread("matrix/bcsstk11.mtx")
B = scipy.io.mmread("matrix/bcsstm11.mtx")

contour = Ellipse(real=200, imag=0.3, shift=900)
option = {'l': 8, 'm': 2, 'n': 12}

comm = MPI.COMM_WORLD
pyss.mpi.solve(A, B, contour, option, comm)
