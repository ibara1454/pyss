#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyss.util.match import replace_attr_if_match
from pyss.algorithm import *
from pyss.helper.generator import random_source_matrix
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


__mapper = {
    'solver': [
        ('linsolve', linsolve)
    ],
    'source': [
        ('random', random_source_matrix)
    ],
    'quadrature': [
        ('trapezoid', 1),
        ('simpson', 2),
        ('simpson3/8', 3),
        ('boole', 4)
    ],
    'executor': [
        ('process_pool_executor', ProcessPoolExecutor),
        ('thread_pool_executor', ThreadPoolExecutor)
    ]
}


def replace_option(option):
    return replace_attr_if_match(__mapper, option)
