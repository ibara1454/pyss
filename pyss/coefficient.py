import numpy as np
import numpy.linalg


def newton_cotes_coeff(h, n, tr=None):
    """Return array of weights of Newton-Cotes method.

    Parameters
    ----------
    h : float
        Length of each two nearest sampling points.
    n : int
        Number of sampling points. Counted from 0.
    tr : int, optional
        Degree of interpolate polynomial using in Newton-Cotes method.
        `tr` should lower or equal to `n`.

    Returns
    -------
    ws : ndarray
        Array of weights of Newton-Cotes method with (`n`) sampling points
    """
    if tr is None:
        tr = n
    a_i = np.arange(n) * h
    b_i = lambda i: ((n-1)*h) ** i / i
    A = np.vstack(a_i ** i for i in range(tr))
    b = np.array([b_i(i+1) for i in range(tr)]).T
    # using least square solver instead of multiplying pseudo inverse
    x, res, rank, sigv = np.linalg.lstsq(A, b)
    return x


def composite_newton_cotes_coeff(h, n, cmpstn, tr=None, contour=False):
    """Return array of weights of composite Newton-Cotes method.

    Parameters
    ----------
    h : float
        Length of each two nearest sampling points.
    n : int
        Number of sampling points of non-composite Newton-Cotes method in this
        composite method. Counted from 0.
        `n` should be a factor of `cmpstn`.
    cmpstn : int
        Number of all sampling points in this composite method. Counted from 0.
        `cmpstn` should be a multiple of `n`.

    Returns
    -------
    ws : ndarray
        Array of weights of Newton-Cotes method with `cmpstn` sampling points
    """
    if contour is False and (cmpstn-1) % (n-1) != 0:
        raise ValueError
    if contour is True and cmpstn % (n-1) != 0:
        raise ValueError
    ws = np.zeros(cmpstn).T
    basews = newton_cotes_coeff(h, n, tr)
    loops = int((cmpstn if contour is True else cmpstn-1) / (n-1))

    begin = 0
    for l in range(0, loops):
        ws = __add_base_coeff_to_composite_coeff(ws, basews, begin)
        begin = begin + (n-1)
    return ws


def __add_base_coeff_to_composite_coeff(cmpstws, basews, begin):
    n = cmpstws.size
    for i in range(0, basews.size):
        cmpstws[(begin+i) % n] = cmpstws[(begin+i) % n] + basews[i]
    return cmpstws


def stable_newton_cotes_coeff(h, n, tr):
    """Return array of weights of Stable Newton-Cotes method.

    Parameters
    ----------
    h : float
        Length of each two nearest sampling points.
    n : int
        Number of sampling points. Counted from 0.
    tr : int
        Degree of interpolate polynomial using in Newton-Cotes method.
        `tr` should lower or equal to `n`.

    Returns
    -------
    ws : ndarray
        Array of weights of Newton-Cotes method with (`n`+1) sampling points
    """
    a_1 = np.linspace(start=0, stop=n*h, num=n+1, endpoint=True)
    b_i = lambda i: (n*h) ** i / i
    A = np.vstack(a_1 ** i for i in range(tr+1))
    b = np.array([b_i(i+1) for i in range(tr+1)]).T
    return np.linalg.pinv(A).dot(b)


def composite_stable_newton_cotes_coeff(h, n, tr, cmpstn, contour=False):
    """Return array of weights of composite stable Newton-Cotes method.

    Parameters
    ----------
    h : float
        Length of each two nearest sampling points.
    n : int
        Number of sampling points of non-composite Newton-Cotes method in this
        composite method. Counted from 0.
        `n` should be a factor of `cmpstn`.
    tr : int
        Degree of interpolate polynomial using in Newton-Cotes method.
        `tr` should lower or equal to `n`.
    cmpstn : int
        Number of all sampling points in this composite method. Counted from 0.
        `cmpstn` should be a multiple of `n`.

    Returns
    -------
    ws : ndarray
        Array of weights of Newton-Cotes method with (`cmpstn`+1) sampling
        points.
    """
    xs = np.zeros(cmpstn+1).T
    basexs = stable_newton_cotes_coeff(h, n, tr)  # vector of length n+1
    for i in range(0, cmpstn, n):
        xs[i:i+n+1] = xs[i:i+n+1] + basexs
    return xs
