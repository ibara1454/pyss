#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils import setup, find_packages

setup(
    name='pyss',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'scipy>=1,<2'
    ],
    description='Eigen solver Sakurai-Sugiura method',
    author='Ibara Takanashi',
    author_email='ibara1454@gmail.com',
    url='',
    license='MIT',
    classifiers=[]
)
