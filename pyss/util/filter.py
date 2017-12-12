#! /usr/bin/env python
# -*- coding: utf-8 -*-


def eig_pair_filter(x, v, func):
    indexes = func(x)
    return x[indexes], v[:, indexes]
