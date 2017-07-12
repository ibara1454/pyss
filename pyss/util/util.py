#! /usr/bin/env python
# -*- coding: utf-8 -*-


def count_iterable(it):
    """
    Returns the length of iterable object

    Parameters
    ----------
    it : iterable
        Iterable object.

    Returns
    -------
    n : int
        length of `it`.
    """
    return sum(1 for i in it)


def eig_pair_filter(x, v, func):
    indexes = func(x)
    return x[indexes], v[:, indexes]
