#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta


class Separator(metaclass=ABCMeta):
    def __call__(a, comm):
        """
        Parameters
        ----------
        a : (N, M) array_like
        comm : MPI communicator

        Returns
        -------
        ws : ndarray
        """
        pass


class HSeparator(Separator):
    """
    A horizontal separator.
    """
    def __call__(a, comm):
        pass


class IdentitySeparator(Separator):
    def __call__(a, comm):
        return a
