#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

print(find_packages("pyss"))

setup(
    name='pyss',
    version='0.1.0',
    packages=find_packages('pyss'),
    package_dir={'': 'pyss'},
    install_requires=[
        'numpy>=1.10.0,<2'
    ],
    # need ldlt decomposition, on future release scipy 1.1
    dependency_links=['git+ssh://git@github.com/scipy/scipy.git'],
    extras_require={
        "mpi": ["mpi4py>=2.0.0,<3"]
    },
    test_suite='test',
    description='Eigen solver Sakurai-Sugiura method',
    author='Ibara Takanashi',
    author_email='ibara1454@gmail.com',
    url='https://github.com/ibara1454/pyss',
    license='MIT',
    classifiers=[]
)
