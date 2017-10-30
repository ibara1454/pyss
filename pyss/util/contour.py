#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy


class Curve:
    """
    A counterclockwise jordan curve, which is a path in complex plane.

    An instance of Curve contains the imformation of such contour path.
    """
    def __init__(self, domain, func, df=None, step=2**-20):
        """
        Parameters
        ----------
        func : function :: float -> complex
            The counterclockwise jordan curve with `domain`.
            That means `func` must be continues and maps `domain` onto a
            counterclockwise jordan curve on complex plane.
        df : function :: float -> complex, option
            The derivative of `func` with `domain`. Default is None.
            If `df` is None, it will automatically generated by using numerical
            differentiation (symmetric difference quotient).
        step : float, optional
            The step size is performed to compute the numerical differentiation
            of `func`. Default is 1e-4.
            If `df` is given, then no `step` will be used in this function.
            Else, `step` will be the step size in numerical differentiation.
        """
        self._domain = domain
        self.__func = func
        if df is not None:
            self.__df = df
        else:
            self.__df = self.__generate_df(step)

    @property
    def domain(self):
        return self._domain

    @property
    def domain_begin(self):
        return self._domain[0]

    @property
    def domain_end(self):
        return self._domain[1]

    @property
    def domain_length(self):
        return self.domain_end - self.domain_begin

    @property
    def func(self):
        return self.__func

    @property
    def df(self):
        return self.__df

    @property
    def center(self):
        """
        Returns the center of jordan curve.

        Assume the center of jordan curve is the center of line passing through
        f(begin) and f(half)
        """
        # TODO: use arithmetic mean of n points instead
        begin = self.domain_begin
        half = self.domain_end / 2
        return (self.func(begin) + self.func(half)) / 2

    def is_inside(self, point):
        pass
        # TODO: implementation
        # use winding number algorithm
        # http://www.nttpc.co.jp/technology/number_algorithm.html

    def __generate_df(self, h):
        """
        Generates derivative of given function using numerical differentiation.

        Uses 4-points symmetric difference quotient to calculate the
        derivative.

        Parameters
        ----------
        h : float
            Step size in numerical differentiation.

        Returns
        -------
        df : function :: float -> complex
            The derivative of `func` which is performed by numerical
            differentiation.
        """
        h2 = h * 2
        h12 = h * 12
        f = self.func
        return lambda x: (f(x - h2) - 8 * f(x - h) + 8 * f(x + h) - f(x + h2)) / h12


#  ==============================
#          Helper classes
#  ==============================
class Ellipse(Curve):
    """
    A counterclockwise jordan curve on complex plane with elliptical shape.
    """
    def __init__(self, real, imag, shift=0.0, rot=0.0):
        """
        Generates the ellipse by given both axises lying on real axis and
        imaginary axis. You could assign a shift value to move the ellipse
        away from origin. And also you can assign a rotation value to
        rotate the ellipse to any degree.

        Parameters
        ----------
        real : float
            The half length of axis on real axis.
        imag : float
            The half length of axis on imaginary axis.
        shift : float or complex
            The shift coordinate of origin.
        rot : float
            Angle of rotation, rot is in the value of [0, 2pi).
        """
        self._center = shift
        self.__major = real if real > imag else imag
        self.__minor = imag if real > imag else real
        trans = lambda x: shift + numpy.exp(1j * rot) * x

        c = numpy.abs(real ** 2 - imag ** 2) ** 0.5
        foci = (-c + 0j, c + 0j) if real > imag else (0 - c * 1j, 0 + c * 1j)
        self.__foci = tuple(trans(x) for x in foci)

        func = lambda x: real * numpy.cos(x) + 1j * imag * numpy.sin(x)
        df = lambda x: - real * numpy.sin(x) + 1j * imag * numpy.cos(x)
        super().__init__(domain=(0, 2 * numpy.pi)
                         func=lambda x: trans(func(x)),
                         df=lambda x: numpy.exp(1j * rot) * df(x))

    @property
    def center(self):
        return self._center

    def is_inside(self, pt):
        """
        Return the boolean (array) value whether each point is inside this
        contour.

        Parameters
        ----------
        pt : array_like
            Input array.

        Returns
        -------
        bmaps : boolean array
            The result of each pt is inside this contour or not.
        """
        c0, c1 = self.__foci
        # Sum of length of (c0 to pt) and (c1 to pt)
        length = numpy.abs(c0 - pt) + numpy.abs(c1 - pt)
        return length < self.__major * 2


class Circle(Ellipse):
    """
    A counterclockwise jordan curve on complex plane with round shape.
    """
    def __init__(self, center, radius):
        """
        Parameters
        ----------
        center : complex
            The center of circle.
        radius : float or complex
            The radius of circle.
        """
        self.__radius = radius
        super().__init__(real=radius, imag=radius, shift=center)

    @property
    def radius(self):
        return self.__radius

    def is_inside(self, pt):
        """
        Return the boolean (array) value whether each point is inside this
        contour.

        Parameters
        ----------
        pt : array_like
            Input array.

        Returns
        -------
        bmaps : boolean array
            The result of each pt is inside this contour or not.
        """
        return numpy.abs(self.center - pt) < self.radius


def inside_filter(x, v, contour):
    index_inside = contour.is_inside(x)
    return x[index_inside], v[:, index_inside]
