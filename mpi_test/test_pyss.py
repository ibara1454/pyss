import scipy.io
import numpy
from pyss.util.contour import Circle, Ellipse

A = scipy.io.mmread("matrix/cage4.mtx")
B = scipy.sparse.eye(9)

contour = Ellipse(real=200, imag=0.3, shift=900)


def create_source(x, y):
    return numpy.eye(x, y)

option = {'l': 2, 'm': 1, 'n': 12, 'source': create_source}
