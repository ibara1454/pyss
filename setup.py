#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


# def load_requires_from_file(filepath):
#     with open(filepath) as fp:
#         return [pkg_name.strip() for pkg_name in fp.readlines()]


# def load_links_from_file(filepath):
#     res = []
#     with open(filepath) as fp:
#         for pkg_name in fp.readlines():
#             if pkg_name.startswith("-e"):
#                 res.append(pkg_name.split(" ")[1])
#     return res

setup(
    name='pyss',
    version='0.1.0',
    packages=find_packages('pyss'),
    package_dir={'': 'pyss'},
    install_requires=[
        'numpy>=1.10.0,<2',
        'scipy'
    ],
    # need ldlt decomposition, on future release of scipy 1.1
    dependency_links=['https://github.com/scipy/scipy.git@master#egg=scipy-0'],
    # install_requires=load_requires_from_file("requirements.txt"),
    # dependency_links=load_links_from_file("requirements.txt"),
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
