#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestLoader, TextTestRunner

path = '.'

if __name__ == '__main__':
    path = '.'
    loader = TestLoader()
    test = loader.discover(path)
    runner = TextTestRunner(verbosity=2)
    runner.run(test)
