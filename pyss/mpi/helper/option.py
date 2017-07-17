#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyss.util.match import replace_if_match
from pyss.mpi.algorithm import *
from pyss.helper.source import *


__solver_mapper = [('linsolve', linsolve)]

__source_mapper = [('random', random_source_matrix)]


def replace_source(source):
    return replace_if_match(__source_mapper, source)


def replace_solver(solver):
    return replace_if_match(__solver_mapper, solver)
