#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='pyss',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.10.0,<2',
        'scipy>=0.18.0,<1',
        'attrdict==2.0.0'
    ],
    extras_require={
        "mpi": ["mpi4py>=2.0.0,<3"]
    },
    description='Eigen solver Sakurai-Sugiura method',
    author='Ibara Takanashi',
    author_email='ibara1454@gmail.com',
    url='https://github.com/ibara1454/pyss',
    license='MIT',
    classifiers=[]
)
